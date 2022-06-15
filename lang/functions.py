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

"""Functions for use when defining expressions.

Common functions that one might want to apply to Expressions. Currently minimal
set needed for some simple tests.
"""

from typing import Any, Callable, Optional, Tuple, Protocol

from metalang.lang import expr


class LazyFn(Protocol):

  def __call__(self, *args: expr.Expr, **kwargs: expr.Expr) -> expr.Expr:
    ...


def wrap(fn: Callable[..., Any],
         spec: Optional[expr.ExprSpec] = None) -> LazyFn:
  """Wraps a standard function into a lazy function.

  This is a little more dangerous than explicitly returning a make_bound
  expression with a fixed function (as done below for make_tuple) since it
  doesn't check that the arguments are correct until evaluation time.

  Args:
    fn: The callable that is being wrapped.
    spec: The spec to provide for the functions output. By default this will
      attempt to infer the resultant spec by applying the function to the spec
      objects. If they do not support the necessary operations this will need to
      be provided DefaultExprSpec should always be a valid fallback.

  Returns:
    An expression for the lazy result of the wrapped function.
  """

  @expr.promote_fn
  def wrapped(*args: expr.Expr, **kwargs: expr.Expr) -> expr.Expr:
    return expr.Expr.make_bound(
        name=fn.__name__, impl=fn, args=args, kwargs=kwargs, spec=spec)

  return wrapped


def make_tuple(*args: expr.Expr) -> expr.Expr[Tuple[Any, ...]]:
  def tup(*args: Any) -> Tuple[Any, ...]:
    return tuple(args)

  return expr.Expr.make_bound(
      name="tuple", impl=tup, args=args, spec=expr.TensorExprSpec((len(args),)))

