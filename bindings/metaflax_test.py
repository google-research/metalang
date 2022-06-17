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
"""Tests for flax."""

from absl.testing import absltest
import flax.linen as nn
import jax
import jax.numpy as jp

from metalang import lang
from metalang.bindings import metaflax


def make_mlp():
  return nn.Sequential(
      [nn.Dense(10), nn.relu,
       nn.Dense(10), nn.relu,
       nn.Dense(10)])


class FlaxTest(absltest.TestCase):

  def test_module_invocation(self):
    module = lang.required("module")
    train = metaflax.DatasetVar("train")
    rng = metaflax.RngVar("params")
    batch = train.next()
    x, y = batch[0], batch[1]
    params = metaflax.init_fn(module, init=True)(rng.next(), x)
    out, params = metaflax.apply_fn(module)(params, x)
    dist = (out - y)**lang.const(2)
    # jp.sum isn't defined for ExprSpecs so we must override and provide a
    # return value spec (DefaultExprSpec).
    score = lang.wrap(jp.sum, lang.DefaultExprSpec())(dist)
    score_mlp = score.partial({module: make_mlp()})

    ## EXEC
    state = score_mlp.initialize({
        rng: [jax.random.PRNGKey(123)],
        train: [[jp.arange(10), jp.arange(10)]]
    })
    result = score_mlp.eval({**state, train: [[jp.ones(10), jp.zeros(10)]]})
    self.assertIsInstance(result, jp.DeviceArray)
    self.assertEqual(result.dtype, jp.float32)

  def test_module_wrapping(self):
    a = lang.var(1.2, "a")
    b = lang.var(2.3, "b")
    c = lang.required("c")
    module = metaflax.ExprModule(a + b + c)
    params = module.init(jax.random.PRNGKey(123), c=3.4)
    self.assertEqual(module.apply(params, c=3.5), 7)
    self.assertDictEqual(params.unfreeze(),
                         {"params": {
                             "state": {
                                 "a": 1.2,
                                 "b": 2.3,
                             }
                         }})

  def test_rng_advancing(self):
    rng = metaflax.RngSequence(123)
    union = set()
    iters = 5
    samples_per = 10
    for _ in range(iters):
      # Sum is obviously not a great hash here, but it is good enough to prove
      # we aren't generating collisions willy nilly.
      union |= {int(jp.sum(rng[i])) for i in range(samples_per)}
      rng = rng.advance()
    self.assertLen(union, iters * samples_per)


if __name__ == "__main__":
  absltest.main()
