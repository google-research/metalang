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

"""Useful expression constants."""

from typing import Any, Callable
from metalang.lang import expr

TRUE = expr.const(True)
FALSE = expr.const(False)
NONE = expr.const(None)


def constant_fn(return_value: Any) -> expr.Expr[Callable[..., Any]]:
  """Produces an expression that evaluates to a constant function.

  This is primarily useful as a placehold for expression functions that have not
  been defined yet (especially when only needed for initialization).

  Args:
    return_value: The value that will be returned by the wrapped fn.
  Returns:
    A const expression that evaluates to a function returning return_value.
  """
  def fn(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return return_value
  return expr.const(fn, f"const_fn<{return_value}>")
