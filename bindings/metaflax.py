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
"""Flax support for metalang.

This is mostly convenience sugar for use with flax.
"""

import dataclasses as dc
from typing import Any, Callable, Optional, Tuple, Sequence

import flax
import jax
from jax.tree_util import register_pytree_node_class

from metalang.bindings import types
from metalang import lang

# TODO: Support dict datasets.

# Chosen by fair dice roll.
_RNG_DEFAULT_SEED = 726883474


class DatasetVar(lang.IteratedVar):

  def __init__(self, name: str, shape: Tuple[int, ...] = tuple()):
    spec = lang.TensorExprSpec(shape) if len(shape) else lang.DefaultExprSpec()
    super().__init__(name, spec)


class RngVar(lang.IteratedVar):
  pass


@register_pytree_node_class
class RngSequence(Sequence):
  """Rng that supports sequence-like lookup.

  This can be used as a value for rng IteratedVars within an Expr.eval.
  """

  def __init__(self,
               seed: int = _RNG_DEFAULT_SEED,
               prng: Optional[types.PRNGKey] = None):
    if prng is not None:
      self._prng = prng
    else:
      self._prng = jax.random.PRNGKey(seed)

  def __len__(self):
    raise TypeError("RngSequence has no length (infinite).")

  def __getitem__(self, s: int) -> types.PRNGKey:
    return jax.random.fold_in(self._prng, s)

  def advance(self) -> "RngSequence":
    return RngSequence(prng=jax.random.split(self._prng, 1)[0])

  def tree_flatten(self) -> Tuple[Tuple[Any, ...], Any]:
    """Adds jax PyTree serialization support for this type."""
    return ((self._prng,), None)

  @classmethod
  def tree_unflatten(cls, aux_data: Any, children: Tuple[Any]) -> "RngSequence":
    """Adds jax PyTree deserialization support for this type."""
    del aux_data
    return RngSequence(prng=children[0])


def init_fn(module: lang.Expr[flax.linen.Module],
            name: Optional[str] = None,
            init: bool = True) -> Callable[..., lang.Expr[types.Params]]:
  """Makes a params Expr constructor for the provided module Expr.

  Args:
    module: The module for which to create params.
    name: The name of the resultant Expr (by default module.name + '_params'.
    init: True if the returned Expr should initialize with the module.init fn.
      Otherwise it will evaluate to the module.init fn.

  Returns:
    A constructor for the module's params which produces exprs either evaluating
    to the modules params, or initializing to it per init argument when called
    with the module init arguments.
  """
  if not name:
    name = module.name + "_params"

  def fn(*args: lang.Expr, **kwargs: lang.Expr) -> lang.Expr[types.Params]:
    init_expr = module.obj.init(*args, **kwargs)
    if init:
      return lang.Expr.make_unbound(name=name, init=init_expr)
    return init_expr

  return fn


def apply_fn(module: lang.Expr[flax.linen.Module]):
  """Creates a function for invoking the module's apply."""

  def fn(params: lang.Expr[types.Params], *args: lang.Expr,
         **kwargs: lang.Expr) -> lang.Expr:
    # We swap out the spec here to support unpacking (assumes this is "mutable")
    # TODO: Figure out a better way to support unpacking and obviate.
    return dc.replace(
        module.obj.apply(params, *args, **kwargs),
        spec=lang.TensorExprSpec((2,)))

  return fn


class ModuleExpr:
  """Represents a flax module for use with metalang.

  This is similar to a standard Expr[Module] except that it provides convenience
  accessors to the init and apply methods of the wrapped Module. It also
  provides convenience controls for how the results of init get wrapped.
  """

  def __init__(self,
               module: lang.Expr[flax.linen.Module],
               method: lang.Expr[Optional[Callable[..., Any]]] = lang.NONE,
               variable_params: bool = True):
    """Constructs a new Module wrapper.

    Args:
      module: The Expr representing a flax module to wrap.
      method: The (optional) method to invoke on the module. Defaults to None.
      variable_params: If true, the params returned by init will be a variable
        that initializes with module.init (rather than an expression that
        evaluates to the module.init value).
    """
    self.expr = module
    self.method = method
    self.variable_params = variable_params

  def init(self, *args: lang.Expr,
           **kwargs: lang.Expr) -> lang.Expr[types.Params]:
    return init_fn(
        self.expr, init=self.variable_params)(
            *args, **kwargs, method=self.method)

  def apply(self, params: lang.Expr[types.Params], *args: lang.Expr,
            **kwargs: lang.Expr) -> lang.Expr:
    return apply_fn(self.expr)(params, *args, method=self.method, **kwargs)


class ExprModule(flax.linen.Module):
  """Expr wrapper that provides a flax module interface."""

  expr: lang.Expr

  @flax.linen.compact
  def __call__(self, *args, **kwargs) -> Any:
    expr_args = self.expr.arg_dict()
    ordered = dict(zip(expr_args.values(), args))
    named = {expr_args[k]: v for k, v in kwargs.items()}
    env = {**ordered, **named}
    if len(env) != len(ordered) + len(named):
      raise SyntaxError("Non keyword arg after keyword arg")

    def init(env):
      return {k.name: v for k, v in self.expr.initialize(env).items()}

    state = self.variable("params", "state", init, env)
    expr_state = {expr_args[k]: v for k, v in state.value.items()}
    return self.expr.eval({**env, **expr_state})
