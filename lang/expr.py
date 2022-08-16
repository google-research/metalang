# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A meta language for defining flax pipelines."""
import collections
import dataclasses as dc
import functools
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, TypeVar, Union

import chex
import immutabledict
from jax.tree_util import register_pytree_node_class

T = TypeVar('T')

Env = Dict[Union['Expr', 'IteratedVar', 'IteratedState'],
           Union[Any, 'IteratedVal', int]]


def _env_split(
    env: Env) -> Tuple[Dict['Expr', Any], Dict['IteratedVar', 'IteratedVal']]:
  """Split the environment into Expr and Iterated assignments."""
  exprs = {}
  iters = {}
  for k, v in env.items():
    if isinstance(k, Expr):
      exprs[k] = v
    elif isinstance(k, IteratedVar):
      if not isinstance(v, IteratedVal):
        raise TypeError(f'IteratedVar {k.name} must be assigned iterator value')
      iters[k] = v
    else:
      raise TypeError('Keys must only be Exprs or IteratedVars.')
  return exprs, iters


class Required:
  pass


class ExprIter(Iterator):
  """Expr iterator, mostly used for assignment expansion."""

  def __init__(self, expr: 'Expr', length: Optional[int] = None):
    self.expr = expr
    self.length = length
    self.index = 0

  def __next__(self):
    self.index += 1
    if self.length and self.index > self.length:
      raise StopIteration
    return self.expr[self.index - 1]


@register_pytree_node_class
class ExprSpec:
  """Base spec implementation.

  ExprSpecs provide metadata (mostly shape data) about the value that an Expr
  will evaluate to. Primarily they provide an interface for accessing the length
  and keys available on an Expr's value. This is primarily used to enable more
  robust unpacking assignment of Exprs. ExprSpecs are generally optional and by
  default a DefaultExprSpec is usually returned which does not provide any
  meaningful shape data. When needed this behavior can be ammended to support
  correct upacking.
  """

  def __len__(self):
    raise NotImplementedError()

  def keys(self) -> Iterable[Any]:
    raise NotImplementedError()

  def __getitem__(self, k: Any) -> 'ExprSpec':
    raise NotImplementedError()

  def get(self, name: str) -> 'ExprSpec':
    raise NotImplementedError()

  def __call__(self, *args, **kwargs) -> 'ExprSpec':
    raise NotImplementedError()

  # Default operators for to support basic shape inference.
  def __add__(self, other: 'ExprSpec') -> 'ExprSpec':
    return self

  def __sub__(self, other: 'ExprSpec') -> 'ExprSpec':
    return self

  def __mul__(self, other: 'ExprSpec') -> 'ExprSpec':
    return self

  def __matmul__(self, other: 'ExprSpec') -> 'ExprSpec':
    return self

  def __truediv__(self, other: 'ExprSpec') -> 'ExprSpec':
    return self

  def __floordiv__(self, other: 'ExprSpec') -> 'ExprSpec':
    return self

  def __mod__(self, other: 'ExprSpec') -> 'ExprSpec':
    return self

  def __divmod__(self, other: 'ExprSpec') -> 'ExprSpec':
    return self

  def __pow__(self,
              other: 'ExprSpec',
              modulo: Optional['ExprSpec'] = None) -> 'ExprSpec':
    return self

  def __lshift__(self, other: 'ExprSpec') -> 'ExprSpec':
    return self

  def __rshift__(self, other: 'ExprSpec') -> 'ExprSpec':
    return self

  def __and__(self, other: 'ExprSpec') -> 'ExprSpec':
    return self

  def __xor__(self, other: 'ExprSpec') -> 'ExprSpec':
    return self

  def __or__(self, other: 'ExprSpec') -> 'ExprSpec':
    return self

  def __neg__(self) -> 'ExprSpec':
    return self

  def __pos__(self) -> 'ExprSpec':
    return self

  def __abs__(self) -> 'ExprSpec':
    return self

  def __invert__(self) -> 'ExprSpec':
    return self

  def tree_flatten(self) -> Tuple[Tuple[Any, ...], Any]:
    """Adds jax PyTree serialization support for this type."""
    return (tuple(), None)

  @classmethod
  def tree_unflatten(cls, aux_data: Any, children: Tuple[Any]) -> 'ExprSpec':
    """Adds jax PyTree deserialization support for this type."""
    del aux_data, children
    return ExprSpec()


@register_pytree_node_class
class TensorExprSpec(ExprSpec):
  """Spec for statically shaped tensor-like objects."""

  def __init__(self, shape: Tuple[int, ...]):
    self.shape = shape

  def __len__(self):
    return self.shape[0]

  def keys(self) -> Iterable[Any]:
    return range(len(self))

  def __getitem__(self, k: int) -> ExprSpec:
    assert k < len(self) and k >= 0
    if len(self.shape) == 1:
      return DefaultExprSpec()
    return TensorExprSpec(self.shape[1:])

  def tree_flatten(self) -> Tuple[Tuple[Any, ...], Any]:
    """Adds jax PyTree serialization support for this type."""
    return ((self.shape,), None)

  @classmethod
  def tree_unflatten(cls, aux_data: Any,
                     children: Tuple[Any]) -> 'TensorExprSpec':
    """Adds jax PyTree deserialization support for this type."""
    del aux_data
    return TensorExprSpec(*children)


@register_pytree_node_class
class DefaultExprSpec(ExprSpec):
  """Trivial Expr spec that is permissive, but uninformed."""

  def __len__(self):
    return 0

  def keys(self) -> Iterable[Any]:
    return set()

  def __getitem__(self, k: Any) -> ExprSpec:
    return DefaultExprSpec()

  def get(self, name: str) -> ExprSpec:
    return DefaultExprSpec()

  def __call__(self, *args, **kwargs) -> ExprSpec:
    return DefaultExprSpec()

  def tree_flatten(self) -> Tuple[Tuple[Any, ...], Any]:
    """Adds jax PyTree serialization support for this type."""
    return (tuple(), None)

  @classmethod
  def tree_unflatten(cls, aux_data: Any,
                     children: Tuple[Any]) -> 'DefaultExprSpec':
    """Adds jax PyTree deserialization support for this type."""
    del aux_data, children
    return DefaultExprSpec()


@register_pytree_node_class
class PrototypeExprSpec(ExprSpec):
  """A expr spec for a prototype object.

  Given a concrete object, will return a spec for that object.
  """

  def __init__(self, prototype: Any):
    self.prototype = prototype

  def __len__(self):
    return len(self.prototype)

  def keys(self) -> Iterable[Any]:
    if hasattr(self.prototype, 'keys'):
      return self.prototype.keys()
    return range(len(self.prototype))

  def __getitem__(self, k: Any) -> 'PrototypeExprSpec':
    return PrototypeExprSpec(self.prototype[k])

  def get(self, name: str) -> 'PrototypeExprSpec':
    return PrototypeExprSpec(getattr(self.prototype, name))

  def __call__(self, *args, **kwargs) -> ExprSpec:
    return DefaultExprSpec()

  def tree_flatten(self) -> Tuple[Tuple[Any, ...], Any]:
    """Adds jax PyTree serialization support for this type."""
    return ((self.prototype,), None)

  @classmethod
  def tree_unflatten(cls, aux_data: Any,
                     children: Tuple[Any]) -> 'PrototypeExprSpec':
    """Adds jax PyTree deserialization support for this type."""
    del aux_data
    return PrototypeExprSpec(*children)


@register_pytree_node_class
class FnPrototypeExprSpec(ExprSpec):
  """A expr spec for a function that returns a fixed object shape."""

  def __init__(self, prototype: Any):
    self.prototype = prototype

  def __call__(self, *args, **kwargs) -> PrototypeExprSpec:
    return PrototypeExprSpec(self.prototype)

  def tree_flatten(self) -> Tuple[Tuple[Any, ...], Any]:
    """Adds jax PyTree serialization support for this type."""
    return ((self.prototype,), None)

  @classmethod
  def tree_unflatten(cls, aux_data: Any,
                     children: Tuple[Any]) -> 'FnPrototypeExprSpec':
    """Adds jax PyTree deserialization support for this type."""
    del aux_data
    return FnPrototypeExprSpec(*children)


class ExprAttrRef:
  """Accessor object for getting attributes of an Expr lazily.

  This effectively shifts the namespace for attributes from Expr to this type so
  that there are fewer name collisions when using the more natural .attribute
  access.
  """

  def __init__(self, expr: 'Expr'):
    # This is given an intentionally long name so it is less likely to conflict.
    self._expr_on_which_to_get000 = expr

  def __getattr__(self, name):
    return self._expr_on_which_to_get000.get(name)


def promote(val: Any) -> 'Expr':
  return val if isinstance(val, Expr) else const(val)


def promote_args(
    args: Tuple[Any],
    kwargs: Dict[str, Any]) -> Tuple[Tuple['Expr'], Dict[str, 'Expr']]:
  return tuple([promote(v) for v in args
               ]), {k: promote(v) for k, v in kwargs.items()}


def promote_fn(fn: Callable[..., Any]) -> Callable[..., 'Expr']:

  @functools.wraps(fn)
  def impl(*args: Any, **kwargs: Any):
    args, kwargs = promote_args(args, kwargs)
    return fn(*args, **kwargs)

  return impl


@register_pytree_node_class
@dc.dataclass(frozen=True, eq=True)
class Expr(Generic[T]):
  """A lazily evaluated function, capable of representing flax modules.

  These can be composed into arbitrarily complex ASTs. This is basically just a
  generic lazy evaluation framework with a few special features to support
  unbound datastreams (for RNGs and datasets).
  """
  # Either impl or init must be provided. Init may be set to "Required" for
  # Exprs without impls and no good initialization value.
  name: str
  impl: Optional[Callable[..., T]] = dc.field(default=None, hash=False)
  kwargs: Mapping[str, 'Expr'] = immutabledict.immutabledict()
  args: Tuple['Expr', ...] = tuple()
  # If an init is provided, this Expr is considered "unbound".
  # If this is None, it considered a bound expression and is not expected to
  # have a value provided. If it is a Expr it will be evaluated during calls to
  # initialize and that value (or one like it) will be expected during
  # evaluating calls. If it is required, a value will not be generated for it
  # during calls to initialize, however it is expected to be provided for any
  # evaluating call. These are likely to be meta-parameters that are expected to
  # be bound via a call to Expr.partial prior to usage.
  init: Optional[Union[Required, 'Expr[T]']] = None
  # Iterator key for iterated expressions (rng/datasets).
  iter_uuid: Optional[int] = None
  spec: ExprSpec = dc.field(
      default_factory=DefaultExprSpec, compare=False, hash=False)

  def __post_init__(self):
    if self.iter_uuid is not None:
      assert isinstance(self.init, Required)
    if (self.init is None) == (self.impl is None):
      raise AttributeError('Exactly one of init, impl must be provided')
    if not isinstance(self.kwargs, immutabledict.immutabledict):
      raise TypeError('kwargs must be immutable')
    if not isinstance(self.args, tuple):
      raise TypeError('args must be immutable (tuple)')
    for arg in self.args + tuple(self.kwargs.values()):
      if not isinstance(arg, Expr):
        raise TypeError(
            f"Arguments must be expressions {self.name}'s argument {arg} is type {type(arg)}. Call .next() for Iterated holders, or wrap as an Expr with metalang.const for python literals."
        )

  def arg_dict(self) -> Dict[str, Union['Expr', 'IteratedVar']]:
    """Gets the unbound arguments and provides a standard naming lookup."""
    iters = {}
    exprs = {}
    for expr in self.unbound:
      if expr.is_iterated:
        if expr.name in exprs:
          raise NameError(f'Redundant expr name {expr.name}')
        if expr.name not in iters:
          iters[expr.name] = IteratedVar(expr.name)
      else:
        if expr.name in exprs:
          if exprs[expr.name] != expr:
            raise NameError(f'Redundant expr name {expr.name}')
        if expr.name in iters:
          raise NameError(f'Redundant expr name {expr.name}')
        exprs[expr.name] = expr
    return {**exprs, **iters}

  def propagate_constants(self) -> 'Expr[T]':
    """Propagates any constants through the AST.

    If any expressions within the AST are fully computable with constants, the
    value is precomputed and frozen. Ensures that re-used expressions
    (expressions occurring as arguments on multiple other expressions) do not
    get fragmented.

    Returns:
      A new expression with all constants propagated through.
    """
    # BFS from root. Find all expressions that are evaluable, evaluate them and
    # replace.
    # TODO: We probably could do a partial restart after a swap rather
    # than a full reset.
    root = self
    q = collections.deque([root])
    while q:
      n = q.pop()
      assert isinstance(n, Expr)
      # If it is already const, no need to propagate.
      if n.is_const:
        continue
      # If we can evaluate, do so.
      if n.can_eval():
        root = root.swap_expr(n, const(n.eval(), str(n)))
        q = collections.deque([root])
      # Otherwise try all children nodes (including init).
      else:
        q.extendleft(n.children)
        # TODO: Figure out why pytype thinks Expr doesn't have init??
        # TODO: File a bug with pytype and add bug link here.
        init = n.init  # pytype: disable=attribute-error
        if isinstance(init, Expr):
          q.appendleft(init)

    return root

  @property
  def children(self) -> List['Expr']:
    args = dict.fromkeys(self.args)
    args.update(dict.fromkeys(self.kwargs.values()))
    return list(args.keys())

  def __str__(self):
    arguments = [str(arg) for arg in self.args]
    arguments += [f'{k}={str(v)}' for k, v in self.kwargs.items()]
    arguments = ','.join(arguments)
    name = self.name
    if self.iter_uuid is not None:
      name = f'{name}_{self.iter_uuid}'
    return f'{name}({arguments})' if arguments else name

  def swap_expr(self, old: 'Expr', new: 'Expr') -> 'Expr[T]':
    """Swaps all instances of old in ast for new.

    This needs to be done carefully and sparingly to avoid generating cycles in
    the AST. New arg should generally only be a const.

    Args:
      old: The old Expr present somewhere in the AST rooted at this Expr.
      new: The new Expr that old should be replaced with.

    Returns:
      A new Expr representing the root of the modified AST.
    """
    if self == old:
      return new

    kwargs = {k: v.swap_expr(old, new) for k, v in self.kwargs.items()}
    args = [v.swap_expr(old, new) for v in self.args]
    init = self.init
    if isinstance(init, Expr):
      init = init.swap_expr(old, new)
    return Expr(
        name=self.name,
        kwargs=immutabledict.immutabledict(kwargs),
        args=tuple(args),
        init=init,
        iter_uuid=self.iter_uuid,
        impl=self.impl)

  def eval(self, env: Optional[Env] = None, **kwargs: Any) -> T:
    """Evaluate the expression given the provided assignments.

    Args:
      env: The environment to use to evaluate the expression.
      **kwargs: Named environmental overrides. Exprs are looked up by argument
        name and given the provided value.

    Raises:
      ValueError: An unbound expression was not provided in the env.
    Returns:
      The resulting value.
    """
    if not env:
      env = {}
    env = env.copy()
    if kwargs:
      env.update(self.named_env(**kwargs))
    return self.eval_and_update(env)

  def eval_and_update(self, env: Env) -> T:
    """Evaluate the expression given the provided (mutable) environment.

    Will update the provided env with a cache of the computed expressions. All
    unbound expression predicates to this expression must be provided in the
    env.

    Args:
      env: The environment to use to evaluate the expression.

    Raises:
      ValueError: An unbound expression was not provided in the env.
    Returns:
      The resulting value.
    """

    if self in env:
      # Handle cached values.
      return env[self]
    if self.is_iterated:
      # Handle uncached iterable values.
      try:
        index = env.get(IteratedState(name=self.name), 0)
        env[IteratedState(name=self.name)] = index + 1
        value = env[IteratedVar(self.name)][index]
      except KeyError as e:
        raise ValueError(f'Missing value for iterable: {self.name}') from e
      except StopIteration as e:
        raise ValueError(f'Ran out of values for iterable: {self.name}') from e
      env[self] = value
      return value
    # Handle standard expressions.
    evaled_args = [arg.eval_and_update(env) for arg in self.args]
    evaled_kwargs = {k: v.eval_and_update(env) for k, v in self.kwargs.items()}
    if self.impl is None:
      raise ValueError(f'Missing value for unbound expression: {self.name}')
    value = self.impl(*evaled_args, **evaled_kwargs)  # pylint: disable=not-callable
    env[self] = value
    return value

  def __call__(self, *args: Any, **kwargs: Any) -> 'Expr':
    args, kwargs = promote_args(args, kwargs)

    def impl(*args, **kwargs):
      return args[0](*args[1:], **kwargs)

    return self.make_bound(
        'call', impl=impl, args=tuple([self]) + args, kwargs=kwargs)

  @property
  def obj(self) -> ExprAttrRef:
    return ExprAttrRef(self)

  def get(self, name: str) -> 'Expr':

    def impl(subject):
      return getattr(subject, name)

    return self.make_bound(
        f'getattr<{name}>', impl=impl, args=[self], spec=self.spec.get(name))

  def wrt(self, *args: Union['Expr', 'IteratedVar'],
          **kwargs: Union['Expr', 'IteratedVar']) -> 'Expr[Callable[..., T]]':
    """Produces a function valued expression for this expression.

    The produced Expression when evaluated will produce a function which given
    the params listed as params will produce the result of evalling this
    function with those params. In other words:

    foo(x, y).eval(env)(a, b) === foo.partial({x:a, y:b}).eval(env)

    !NOTE: be very careful about iterable arguments. Iterables get evaluated and
    bound ONCE when they are eval'd. As such, most of the time you probably want
    this to be a function of any/all iterable types within.

    Args:
      *args: positional args for the generated function expression.
      **kwargs: named argus for the generated function expression.

    Returns:
      An expression that evaluates to a function with the provided args.
    """
    # Construct a unique instance tree - no repeated exprs in the tree.
    # Prune all descendants of exprs in args_set
    # Prune all ancestors of exprs in args_set
    # Compute the minimum cover of the remaining nodes
    #   This can be done by just greedily taking the "oldest" during BFS.
    # The cover nodes' expressions become the arguments to the produced expr.
    its = [it for it in args if isinstance(it, IteratedVar)]
    exprs = [arg for arg in args if isinstance(arg, Expr)]
    kwexprs = {k: v for k, v in kwargs.items() if isinstance(v, Expr)}
    kwits = {k: v for k, v in kwargs.items() if isinstance(v, IteratedVar)}

    args_set = set(exprs).union(kwexprs.values())
    iter_set = set([it.name for it in its
                   ]).union(set([v.name for v in kwits.values()]))
    exprs = set()

    # TODO: Replace MutableAst with a single array indexed by uuid.

    root = _MutableAst(self, uuid=0)
    q = collections.deque([root])
    idx = 1
    while q:
      n = q.pop()
      children = set()
      for child in n.expr.children:
        children.add(_MutableAst(child, parent=n, uuid=idx))
        idx += 1
      n.children = children
      q.extendleft(children)
      if n.expr in args_set or (n.expr.is_iterated and n.expr.name in iter_set):
        n.needed = False
        exprs.add(n)

    ancestors = set()
    descendants = set()
    for node in exprs:
      if node.parent:
        ancestors.add(node.parent)
      descendants.update(node.children)
    while ancestors:
      ancestor = ancestors.pop()
      ancestor.needed = False
      if ancestor.parent:
        ancestors.add(ancestor.parent)
    while descendants:
      descendant = descendants.pop()
      descendant.needed = False
      descendants.update(descendant.children)

    needed_exprs = set()
    q = collections.deque([root])
    while q:
      n = q.pop()
      if n.needed:
        needed_exprs.add(n.expr)
      else:
        # Don't need to include needed node's children.
        q.extendleft(n.children)

    expr_args = tuple(needed_exprs)

    def expr_fn(*expr_values):
      outer_values = dict(zip(expr_args, expr_values))

      def fn(*arg_values, **kwarg_values):
        # TODO: Check behavior of over-written outer_values.
        ast_values = outer_values.copy()
        ast_values.update(dict(zip(args + tuple(kwargs.values()), arg_values)))
        ast_values.update({kwargs[k]: v for k, v in kwarg_values.items()})

        expr_values = dict(
            filter(lambda e: isinstance(e[0], Expr), ast_values.items()))
        iter_values = dict(
            filter(lambda e: isinstance(e[0], IteratedVar), ast_values.items()))
        iter_values = {IteratedVar(k.name): v for k, v in iter_values.items()}
        expr_values.update(iter_values)
        return self.eval(expr_values)

      return fn

    return Expr(name='fn_expr', impl=expr_fn, args=expr_args)

  def named_env(self, **kwargs: Any) -> Env:
    arg_dict = self.arg_dict()
    return {arg_dict[name]: value for name, value in kwargs.items()}

  # TODO: Add support for extracting remapped named.
  def partial(self, env: Optional[Env] = None, **kwargs: Any) -> 'Expr[T]':
    """Partially evaluate the AST with the provided env.

    Will effectively bind all known values and produce a simplified expr. If an
    expr is present on an initializer field it will also get reduced.

    Note that this will cause the exprs of the AST to mutate and you may not be
    able to use old references when providing values!

    Args:
      env: The partially populated environment.
      **kwargs: Named environmental overrides. Exprs are looked up by argument
        name and given the provided value.

    Returns:
      An expression with bound partial environment.
    """
    if not env:
      env = {}
    env = env.copy()
    if kwargs:
      env.update(self.named_env(**kwargs))
    expr_lut, iter_lut = _env_split(env)
    partial = self
    for expr, value in expr_lut.items():
      partial = partial.swap_expr(expr, const(value, expr.name))
    binding_names = set([e.name for e in iter_lut])
    name_exprs = collections.defaultdict(set)
    if binding_names:
      # Search the partial for all Exprs with iterables named in binding names.
      q = collections.deque([partial])
      while q:
        n = q.pop()
        assert isinstance(n, Expr)
        if n.is_iterated and n.name in binding_names:
          name_exprs[n.name].add(n)
        q.extend(n.children)
        init = n.init  # pytype: disable=attribute-error
        if isinstance(init, Expr):
          q.append(init)
      # Provide a stable ordering by sorting by UUID.
      name_exprs = {
          k: sorted(list(v), key=lambda x: x.iter_uuid)
          for k, v in name_exprs.items()
      }
      # Swap and iterate each unique Expr.
      for iter_var, seq in iter_lut.items():
        for expr in name_exprs[iter_var.name]:
          index = env.get(IteratedState(name=expr.name), 0)
          env[IteratedState(name=expr.name)] = index + 1
          bound = const(seq[index], f'{expr.name}_{expr.iter_uuid}')
          partial = partial.swap_expr(expr, bound)
    return partial.propagate_constants()

  @property
  def fn(self) -> Callable[..., T]:
    """Extract a callable function with arguments for each unbound expr.

    Raises:
      NameError: When a name is used ambiguously in the AST.
    """
    args = self.unbound
    arg_names = [arg.name for arg in args if not arg.is_iterated]
    iterated_names = {arg.name for arg in args if arg.is_iterated}
    if len(set(arg_names)) != len(arg_names):
      missing = [n for n in arg_names if arg_names.count(n) > 1]
      raise NameError(f'Duplicate named unbound variables: {missing}')
    both = iterated_names.intersection(arg_names)
    if both:
      raise NameError(f'Name used by iterated and non-iterated: {both}')

    arg_lut = {arg.name: arg for arg in args if not arg.is_iterated}

    def fn(*args, **kwargs):
      if len(args) > len(arg_names):
        raise KeyError('Iterated arguments must be provided by name')
      named_values = dict(zip(arg_names, args))
      named_values.update({k: v for k, v in kwargs.items() if k in arg_lut})
      memo = {arg_lut[k]: v for k, v in named_values.items()}
      iters = {k: v for k, v in kwargs.items() if k not in arg_lut}
      if set(iters.keys()) != iterated_names:
        raise KeyError(
            f'Needed iterators {iterated_names}, got {set(iters.keys())}')
      memo.update({IteratedVar(k): v for k, v in iters.items()})
      return self.eval(memo)

    return fn

  def initialize(self,
                 env: Optional[Env] = None,
                 **kwargs: Any) -> Dict['Expr', Any]:
    """Initializes all unbound expressions to their init value.

    Args:
      env: The values used during evaluation of initial values.
      **kwargs: Named environmental overrides. Exprs are looked up by argument
        name and given the provided value.

    Returns:
      The newly initialized values (in an env dict).
    """
    if env is None:
      env = {}
    env = env.copy()
    if kwargs:
      env.update(self.named_env(**kwargs))
    init_vals = {}
    unevaluated_exprs = set()
    for node in self.unbound:
      init = node.init
      if init is None:
        continue
      if node in env or IteratedVar(node.name) in env:
        # We don't add required values to the init env.
        if not isinstance(init, Required):
          init_vals[node] = env[node]
        continue
      if isinstance(init, Required):
        # This expr may be required for eval, but not for init.
        continue
      elif isinstance(init, Expr):
        if init.can_eval(env):
          init_vals[node] = init.eval(env)
        else:
          unevaluated_exprs.add(node)
      else:
        raise TypeError(f'Illegally typed init {type(init)}.')
    prop_env = env.copy()
    prop_env.update(init_vals)
    evaluated = {None}
    while evaluated:
      evaluated = set()
      for unev in unevaluated_exprs:
        init = unev.init
        assert isinstance(init, Expr)
        prop_env.update(init.initialize(prop_env))
        if init.can_eval(prop_env):
          res = init.eval(prop_env)
          prop_env[unev] = res
          init_vals[unev] = res
          evaluated.add(unev)
      unevaluated_exprs.difference_update(evaluated)

    if unevaluated_exprs:
      raise ValueError(f'Could not eval init exprs {unevaluated_exprs}')

    return init_vals

  def ordered_unbound(self) -> Dict['Expr', None]:
    res = dict()
    if not self.is_bound:
      res[self] = None
    for arg in self.children:
      res.update(arg.ordered_unbound())
    return res

  def can_eval(
      self,
      env_keys: Optional[Set[Union['Expr', 'IteratedVar']]] = None) -> bool:
    """Determines if the expression can be evaluated with the provided env keys.

    Args:
      env_keys: A set containing the Exprs that would be provided to eval.

    Returns:
      True if the expr can be evaluated with the provided variables known.
    """
    if not env_keys:
      env_keys = set()
    if self in env_keys:
      return True
    if self.is_iterated and IteratedVar(self.name) in env_keys:
      return True
    if not self.is_bound:
      return False
    for child in self.children:
      if not child.can_eval(env_keys):
        return False
    return True

  @property
  def is_const(self) -> bool:
    """True if the expression is effectively constant."""
    return self.impl is not None and not self.args and not self.kwargs

  @property
  def is_bound(self) -> bool:
    return self.impl is not None

  @property
  def is_iterated(self) -> bool:
    return self.iter_uuid is not None

  @property
  def unbound(self) -> List['Expr']:
    """Returns the (ordered) list of unbound predicates.

    Note that this does not include any unbound predicates that are only
    required for initialization. For that, use init_predicates. Non-iterated
    expressions will be returned before all iterated expressions in the result
    list.
    """
    return sorted(
        list(self.ordered_unbound().keys()), key=lambda x: x.is_iterated)

  @property
  def is_initializable(self) -> bool:
    return isinstance(self.init, Expr)

  @property
  def initializable_exprs(self) -> List['Expr']:
    return [e for e in self.unbound if e.is_initializable]

  @property
  def is_required(self) -> bool:
    return isinstance(self.init, Required)

  @property
  def required_for_eval(self) -> List['Expr']:
    return [e for e in self.unbound if e.is_required]

  @property
  def required_for_init(self) -> List['Expr']:
    """Returns the list of arguments that must be provided for initialization."""
    initializable = self.initializable_exprs
    predicates = {}
    for expr in initializable:
      init = expr.init
      assert isinstance(init, Expr)
      predicates.update(dict.fromkeys(init.required_for_init))
      predicates.update(dict.fromkeys(init.required_for_eval))
    return list(predicates.keys())

  def iterated_expr_count(self, expr_or_name: Union['IteratedVar[T]',
                                                    str]) -> int:
    """Get the number of iterated var entries needed to evaluate this expr."""
    name = expr_or_name
    if isinstance(expr_or_name, IteratedVar):
      name = expr_or_name.name
    return len(
        {e.iter_uuid for e in self.unbound if e.is_iterated and e.name is name})

  ## Operators

  def __iter__(self) -> ExprIter:
    return ExprIter(self, len(self.spec))

  def __getitem__(self, key: Any) -> 'Expr':
    return Expr(
        name=f'get<{key}>',
        args=(self,),
        impl=lambda res: res[key],
        spec=self.spec[key])

  @promote_fn
  def __add__(self, other: Any) -> 'Expr':
    return self.make_bound('add', lambda a, b: a + b, args=(self, other))

  @promote_fn
  def __radd__(self, other: Any) -> 'Expr':
    return self.make_bound('add', lambda a, b: a + b, args=(other, self))

  @promote_fn
  def __sub__(self, other: Any) -> 'Expr':
    return self.make_bound('sub', lambda a, b: a - b, args=(self, other))

  @promote_fn
  def __rsub__(self, other: Any) -> 'Expr':
    return self.make_bound('sub', lambda a, b: a - b, args=(other, self))

  @promote_fn
  def __mul__(self, other: Any) -> 'Expr':
    return self.make_bound('mul', lambda a, b: a * b, args=(self, other))

  @promote_fn
  def __rmul__(self, other: Any) -> 'Expr':
    return self.make_bound('mul', lambda a, b: a * b, args=(other, self))

  @promote_fn
  def __matmul__(self, other: Any) -> 'Expr':
    return self.make_bound('matmul', lambda a, b: a @ b, args=(self, other))

  @promote_fn
  def __rmatmul__(self, other: Any) -> 'Expr':
    return self.make_bound('matmul', lambda a, b: a @ b, args=(other, self))

  @promote_fn
  def __truediv__(self, other: Any) -> 'Expr':
    return self.make_bound('truediv', lambda a, b: a / b, args=(self, other))

  @promote_fn
  def __rtruediv__(self, other: Any) -> 'Expr':
    return self.make_bound('truediv', lambda a, b: a / b, args=(other, self))

  @promote_fn
  def __floordiv__(self, other: Any) -> 'Expr':
    return self.make_bound('floordiv', lambda a, b: a // b, args=(self, other))

  @promote_fn
  def __rfloordiv__(self, other: Any) -> 'Expr':
    return self.make_bound('floordiv', lambda a, b: a // b, args=(other, self))

  @promote_fn
  def __mod__(self, other: Any) -> 'Expr':
    return self.make_bound('mod', lambda a, b: a % b, args=(self, other))

  @promote_fn
  def __rmod__(self, other: Any) -> 'Expr':
    return self.make_bound('mod', lambda a, b: a % b, args=(other, self))

  @promote_fn
  def __divmod__(self, other: Any) -> 'Expr':
    return self.make_bound('divmod', divmod, args=(self, other))

  @promote_fn
  def __rdivmod__(self, other: Any) -> 'Expr':
    return self.make_bound('divmod', divmod, args=(other, self))

  def __pow__(self, other: Any, modulo: Any = None) -> 'Expr':
    if modulo is not None:
      return self.make_bound(
          'powm', pow, args=(self, promote(other), promote(modulo)))
    return self.make_bound('pow', pow, args=(self, promote(other)))

  def __rpow__(self, other: Any, mod: Any = None) -> 'Expr':
    if mod is not None:
      return self.make_bound(
          'powm', pow, args=(promote(other), self, promote(mod)))
    return self.make_bound('pow', pow, args=(promote(other), self))

  @promote_fn
  def __lshift__(self, other: Any) -> 'Expr':
    return self.make_bound('lshift', lambda a, b: a << b, args=(self, other))

  @promote_fn
  def __rlshift__(self, other: Any) -> 'Expr':
    return self.make_bound('lshift', lambda a, b: a << b, args=(other, self))

  @promote_fn
  def __rshift__(self, other: Any) -> 'Expr':
    return self.make_bound('rshift', lambda a, b: a >> b, args=(self, other))

  @promote_fn
  def __rrshift__(self, other: Any) -> 'Expr':
    return self.make_bound('rshift', lambda a, b: a >> b, args=(other, self))

  @promote_fn
  def __and__(self, other: Any) -> 'Expr':
    return self.make_bound('and', lambda a, b: a & b, args=(self, other))

  @promote_fn
  def __rand__(self, other: Any) -> 'Expr':
    return self.make_bound('and', lambda a, b: a & b, args=(other, self))

  @promote_fn
  def __xor__(self, other: Any) -> 'Expr':
    return self.make_bound('xor', lambda a, b: a ^ b, args=(self, other))

  @promote_fn
  def __rxor__(self, other: Any) -> 'Expr':
    return self.make_bound('xor', lambda a, b: a ^ b, args=(other, self))

  @promote_fn
  def __or__(self, other: Any) -> 'Expr':
    return self.make_bound('or', lambda a, b: a | b, args=(self, other))

  @promote_fn
  def __ror__(self, other: Any) -> 'Expr':
    return self.make_bound('or', lambda a, b: a | b, args=(other, self))

  def __neg__(self) -> 'Expr':
    return self.make_bound('neg', lambda x: -x, args=(self,))

  def __pos__(self) -> 'Expr':
    return self.make_bound('pos', lambda x: +x, args=(self,))

  def __abs__(self) -> 'Expr':
    return self.make_bound('abs', abs, args=(self,))

  def __invert__(self) -> 'Expr':
    return self.make_bound('invert', lambda x: ~x, args=(self,))

  def __bool__(self):
    """You almost certainly don't want to be using an expression as a condition.

    If you find yourself writing:

    if some_expr:
      do_something()

    This will _not_ do what it looks like. some_expr doesn't have a value until
    eval is called so it cannot be conditioned upon. If you intend to check if
    the expr is None, you can still do that with (some_expr is None).

    Raises:
      RuntimeError: Whenever invoked.
    """
    raise RuntimeError('Expression cannot be evaluated as booleans. ' +
                       self.__bool__.__doc__)

  def __lt__(self, other: Any) -> bool:
    return hash(self) < hash(other)

  ## Constructors

  @staticmethod
  def make_unbound(
      name: str,
      init: Union[Callable[..., T], Required, 'Expr[T]'] = Required(),
      spec: ExprSpec = DefaultExprSpec()
  ) -> 'Expr[T]':
    return Expr(name=name, init=init, spec=spec)

  @staticmethod
  def make_bound(name: str,
                 impl: Callable[..., T],
                 args: Iterable['Expr'] = tuple(),
                 kwargs: Optional[Dict[str, 'Expr']] = None,
                 spec: Optional[ExprSpec] = None) -> 'Expr[T]':
    """Convenience constructor for expression construction.

    Converts args and kwargs to the requisit immutable types.
    Args:
      name: The name for the generated expression.
      impl: The implementation function for the generated expression.
      args: The positional arguments for the generated expression.
      kwargs: The named arguments for the generated expression.
      spec: The spec override for the produced expression.

    Returns:
      An unbound expression for the given parameters.
    """
    if kwargs is None:
      kwargs = {}
    if spec is None:
      spec = impl(*[arg.spec for arg in args],
                  **{k: v.spec for k, v in kwargs.items()})
    return Expr(
        name=name,
        args=tuple(args),
        kwargs=immutabledict.immutabledict(kwargs),
        impl=impl,
        spec=spec)

  ## Jaxtree support

  def tree_flatten(self) -> Tuple[Tuple[Any, ...], Any]:
    """Adds jax PyTree serialization support for this type."""
    return ((self.name, self.impl, self.kwargs, self.args, self.init,
             self.iter_uuid, self.spec), None)

  @classmethod
  def tree_unflatten(cls, aux_data: Any, children: Tuple[Any]) -> 'Expr':
    """Adds jax PyTree deserialization support for this type."""
    del aux_data
    return Expr(
        name=children[0],
        impl=children[1],
        kwargs=children[2],
        args=children[3],
        init=children[4],
        iter_uuid=children[5],
        spec=children[6])


def expr_var(initial_expr: Expr[T], name: str) -> Expr[T]:
  """A variable that initializes to a value defined by an expression.

  Mostly useful for testing.

  Args:
    initial_expr: The expression used for initializing the value.
    name: The name of this expression.

  Returns:
    A new expression that is unbound (and initialized with initial_expr).
  """
  return Expr.make_unbound(name=name, init=initial_expr, spec=initial_expr.spec)


def required(name: str, spec: ExprSpec = DefaultExprSpec()) -> Expr:
  return Expr.make_unbound(name=name, init=Required(), spec=spec)


def const(value: T, name: Optional[str] = None) -> Expr[T]:
  """Create a bound constant Expr with the provided value.

  Args:
    value: The value the const will be evaluated to.
    name: The name for this variable (defaults to the str of the value).

  Returns:
    An expression that will evaluate to value.
  """
  if name is None:
    name = str(value)
  return Expr.make_bound(
      name=name, impl=lambda: value, spec=PrototypeExprSpec(value))


def var(initial_value: T, name: str) -> Expr[T]:
  return Expr.make_unbound(
      name=name,
      init=const(initial_value, name + '_init'),
      spec=PrototypeExprSpec(initial_value))


@chex.dataclass
class IteratedState:
  name: str

  def __hash__(self):
    return hash(self.name) ^ hash(self.__class__)


@register_pytree_node_class
class IteratedVar(Generic[T]):
  """Collection of Exprs that are backed by a single iterator assignment.

  This produces a sequence of unique Exprs that can all be addressed via
  assignment of an iterator to this handle.
  """

  def __init__(self, name: str, spec: ExprSpec = DefaultExprSpec()):
    self.name = name
    self.uuid = 0
    self.spec = spec

  def next(self) -> Expr[T]:
    self.uuid += 1
    return Expr(
        name=self.name, init=Required(), iter_uuid=self.uuid, spec=self.spec)

  def __hash__(self):
    return hash(self.name)

  def __eq__(self, other: Any) -> bool:
    return self.name == other.name

  def __lt__(self, other: Any) -> bool:
    return hash(self) < hash(other)

  def tree_flatten(self) -> Tuple[Tuple[Any, ...], Any]:
    """Adds jax PyTree serialization support for this type."""
    return ((self.name, self.uuid, self.spec), None)

  @classmethod
  def tree_unflatten(cls, aux_data: Any, children: Tuple[Any]) -> 'IteratedVar':
    """Adds jax PyTree deserialization support for this type."""
    del aux_data
    ret = IteratedVar(children[0], children[2])
    ret.uuid = children[1]
    return ret


IteratedVal = Sequence


@dc.dataclass(unsafe_hash=True)
class _MutableAst:
  """Helper class for certain AST traversals."""
  expr: Expr = dc.field(compare=False)
  uuid: int = dc.field(compare=True)
  children: Set['_MutableAst'] = dc.field(default_factory=set, compare=False)
  parent: Optional['_MutableAst'] = dc.field(default=None, compare=False)
  needed: bool = dc.field(default=True, compare=False)
