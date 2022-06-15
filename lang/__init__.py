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

"""Meta language api."""

from .consts import constant_fn
from .consts import FALSE
from .consts import NONE
from .consts import TRUE
from .expr import const
from .expr import DefaultExprSpec
from .expr import Env
from .expr import Expr
from .expr import expr_var
from .expr import ExprSpec
from .expr import IteratedVal
from .expr import IteratedVar
from .expr import PrototypeExprSpec
from .expr import required
from .expr import TensorExprSpec
from .expr import var
from .functions import make_tuple
from .functions import wrap
#
#__all__ = (
#    'constant_fn'
#    'FALSE',
#    'NONE',
#    'TRUE',
#    'const',
#    'DefaultExprSpec',
#    'Env',
#    'Expr',
#    'expr_var',
#    'ExprSpec',
#    'IteratedVal',
#    'IteratedVar',
#    'PrototypeExprSpec',
#    'required',
#    'TensorExprSpec',
#    'var',
#    'make_tuple',
#    'wrap',
#)

