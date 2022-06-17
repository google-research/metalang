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

"""Tests for metalang."""
from absl.testing import absltest
import jax
import jax.numpy as jp
from metalang import lang
import numpy as np


def l2(output: lang.Expr, target: lang.Expr) -> lang.Expr:
  return lang.Expr.make_bound(
      name="l2",
      impl=lambda x, y: jp.sum((x - y)**2),
      kwargs={
          "x": output,
          "y": target
      },
      spec=lang.PrototypeExprSpec(1.0))


class LangTest(absltest.TestCase):

  def test_fn_lift_jittable(self):
    a = lang.var(jp.array([1, 2, 3]), "a")
    b = lang.var(jp.array([3, 4, 5]), "b")
    score = l2(a, b)

    init = score.initialize()
    self.assertEqual(score.eval(init), 12)
    self.assertEqual(score.fn(a=jp.array([1]), b=jp.array([3])), 4)

    fn = score.wrt(a).eval({b: jp.array([1, 2])})
    fn = jax.jit(fn)
    self.assertEqual(fn(jp.array([2, 3])), 2)

  def test_expression_printing(self):
    a = lang.var(jp.array([1, 2, 3]), "a")
    b = lang.const(jp.array([2, 3, 4]), "b")
    c = lang.const(3, "c")
    score = l2(a, b)
    bias = score + c + a[0]
    self.assertEqual("add(add(l2(x=a,y=b),c),get<0>(a))", str(bias))

  def test_lifted_grad(self):
    a = lang.var(jp.array([1, 2, 3]), "a")
    b = lang.var(jp.array([2, 3, 4]), "b")
    score = l2(a, a + b)
    g = jax.grad(score.wrt(a).eval({b: jp.array([1., 2., 3.])}))
    # Ensure that a repeated expression in a lifted grad gets properly canceled.
    np.testing.assert_allclose(g(jp.array([3., 4., 5.])), jp.zeros(3))

    g = jax.grad(score.wrt(b).eval({a: jp.array([2., 3., 4.])}))
    np.testing.assert_allclose(
        g(jp.array([1., 2., 3.])), jp.array([2., 4., 6.]))

  def test_basic_iterable(self):
    req = lang.required("dummy").init
    a = lang.Expr("it", iter_uuid=0, init=req)
    b = lang.Expr("it", iter_uuid=1, init=req)
    score = l2(a, b)

    ds = jp.stack([jp.arange(1000), jp.arange(1000) + 1, jp.arange(1000) + 2]).T

    self.assertEqual(score.eval({lang.IteratedVar("it"): ds}), 3)
    # We can swap out assignments and we don't get iteration
    self.assertEqual(
        score.eval({
            a: jp.array([0, 1, 2]),
            lang.IteratedVar("it"): ds
        }), 0)
    # The re-evaluating the same entry does not cause double iteration.
    score = l2(a, a)
    self.assertEqual(score.eval({lang.IteratedVar("it"): ds}), 0)

  def test_iter_lookup(self):
    itervar = lang.IteratedVar("name")
    a = {itervar: "a"}
    itervar.next()
    self.assertEqual(a[itervar], "a")
    self.assertEqual(a[lang.IteratedVar("name")], "a")

  def test_named_iterated(self):
    itervar = lang.IteratedVar("somename")
    a = itervar.next()
    b = itervar.next()
    score = l2(a, b)

    ds = jp.stack([jp.arange(1000), jp.arange(1000) + 1, jp.arange(1000) + 2]).T

    self.assertEqual(score.iterated_expr_count(itervar), 2)
    self.assertEqual(score.iterated_expr_count("somename"), 2)
    # We can evaluate it with a different instance with same name.
    self.assertEqual(score.eval({lang.IteratedVar("somename"): ds}), 3)
    # Or the same name.
    self.assertEqual(score.eval({itervar: ds}), 3)

  def test_iterable_lift(self):
    itervar = lang.IteratedVar("it")
    a = itervar.next()
    b = itervar.next()
    self.assertNotEqual(a, b)
    score = l2(a, b)

    ds = jp.stack([jp.arange(1000), jp.arange(1000) + 1, jp.arange(1000) + 2]).T

    self.assertEqual(score.wrt(itervar).eval()(ds), 3)
    # If you provide direct values you don't need to provide iterators.
    self.assertEqual(
        score.wrt(a).eval({b: jp.array([2, 3, 4])})(jp.array([1, 2, 3])), 3)
    # Or you can mix and match...
    self.assertEqual(score.wrt(a).eval({itervar: ds})(jp.array([1, 2, 3])), 3)

  def test_conditioning_disabled(self):

    def bad_idea():
      x = lang.var(False, "a")
      if x:
        x = lang.const(22, "b")
      return x

    def ok_idea():
      x = lang.var(False, "a")
      if x is not None:
        return ~x
      assert False

    self.assertRaises(RuntimeError, bad_idea)
    ok_idea()

  def test_can_eval(self):
    x = (lang.const(1) + lang.const(3)) * lang.const(2)
    v = lang.var(1, "v")
    y = v + x
    print(v)
    self.assertFalse(v.can_eval())
    self.assertTrue(x.can_eval())
    self.assertTrue(y.can_eval({v}))
    self.assertTrue(y.can_eval({y}))
    self.assertFalse(y.can_eval({x}))
    self.assertFalse(y.can_eval())
    self.assertFalse(v.can_eval({y}))
    self.assertFalse(v.can_eval())

  def test_lazy_init(self):
    x = lang.required("initializer")
    y = lang.expr_var(x(), "y")
    # We can't initialize until we have defined a value for x.
    self.assertRaises(ValueError, y.initialize)
    p = y.partial({x: lambda: 123})
    self.assertDictEqual(p.initialize(), {p: 123})

  def test_constant_propagation(self):
    x = lang.const(3)
    y = lang.const(4)
    z = (x + y)**x
    z = z + lang.const(1)
    self.assertEqual(z.eval(), 344)
    self.assertNotEmpty(z.args)
    c = z.propagate_constants()
    self.assertEqual(c.impl(), 344)
    self.assertEqual(c.eval(), 344)
    self.assertEmpty(c.args)
    self.assertEmpty(c.kwargs)

  def test_partial_eval(self):
    x = lang.required("x")
    y = lang.const(4)
    z = (x + y)**x
    z = z + lang.const(1)
    self.assertEqual(z.eval({x: 3}), 344)
    self.assertNotEmpty(z.args)
    c = z.partial({x: 3})
    self.assertEqual(c.impl(), 344)
    self.assertEqual(c.eval(), 344)
    self.assertEmpty(c.args)
    self.assertEmpty(c.kwargs)

  def test_expr_unpacking(self):
    x = lang.const([1, 2])
    a, b = x
    self.assertEqual(a.eval(), 1)
    self.assertEqual(b.eval(), 2)

  def test_wrt_initialization(self):
    a = lang.required("a")
    b = lang.expr_var(a, "b")
    x = a
    for _ in range(10):
      x += b
    # A should still be unbound for x.wrt(a) because it is required to init.
    params = x.wrt(a).initialize({a: 1})
    self.assertDictEqual(params, {b: 1})

  def test_initialization_expr_partial(self):
    a = lang.required("a")
    b = lang.expr_var(a, "b")
    x = a
    for _ in range(10):
      x += b
    e = x.partial({a: 2})
    params = e.initialize()
    self.assertEqual(e.eval(params), 22)
    self.assertEqual(e.eval(b=3), 32)

  def test_trivial_partial(self):
    a = lang.required("a")
    e = a.partial({a: 123})
    self.assertEqual(e.eval(), 123)

  def test_init_chain(self):
    a = lang.required("a")
    b = lang.expr_var(a, "b")
    c = lang.expr_var(b, "c")
    d = lang.expr_var(c, "d")
    self.assertListEqual(d.required_for_init, [a])
    result = d.initialize({a: 1})
    self.assertDictEqual(result, {d: 1})

  def test_partial_mutation(self):
    a = lang.required("a")
    b = lang.required("b")
    c = a + b
    d = a + c
    f = d.partial({a: 1})
    # b is unchanged by partial eval of a.
    self.assertEqual(f.eval({b: 2}), 4)
    self.assertEqual(f.eval(b=2), 4)
    # c is changed and can no longer be addressed by its old expr.
    self.assertRaises(ValueError, f.eval, {c: 2})
    # There isn't actually a expr named c. It is an implicitly named add node.
    self.assertRaises(KeyError, f.eval, c=2)

  def test_partial_reduces_unbound(self):
    a = lang.required("a")
    b = lang.required("b")
    c = a(b)
    d = c + b
    self.assertCountEqual(d.unbound, [a, b])
    self.assertLen(d.partial({b: 2}).unbound, 1)
    self.assertLen(d.partial({a: lambda x: x}).unbound, 1)
    self.assertEqual(d.partial({b: 2}).eval(a=lambda x: x + 1), 5)

  def test_partial_init_reduces_unbound(self):
    a = lang.required("a")
    b = lang.expr_var(a(lang.const(123)), "b")
    c = a(b)
    d = c + b
    self.assertCountEqual(d.unbound, [a, b])
    self.assertLen(d.partial({b: 2}).unbound, 1)
    self.assertLen(d.partial({a: lambda x: x}).unbound, 1)
    self.assertEqual(d.partial({b: 2}).eval(a=lambda x: x + 1), 5)

  def test_init_swap_expr(self):
    a = lang.required("a")
    b = lang.expr_var(a(lang.const(123)), "b")
    c = a(b)
    d = c + b
    self.assertLen(d.swap_expr(a, lang.const(lambda x: x)).unbound, 1)

  def test_propagate_constants_uniqueness(self):
    a = lang.const(1)
    b = lang.expr_var(lang.const(2) * a, "b")
    c = b + (a * b)
    self.assertLen(c.propagate_constants().unbound, 1)
    c = (a * b) + b
    self.assertLen(c.propagate_constants().unbound, 1)

  def test_init_with_eval_required(self):
    a = lang.var(1.0, "a")
    b = lang.required("b")
    c = a+b
    self.assertEqual({a: 1.0}, c.initialize())

  def test_op_promotion(self):
    a = lang.const(1)
    self.assertEqual((a - 2).eval(), -1)
    self.assertEqual((2 - a).eval(), 1)
    # Pow has some strange edge cases since it takes 3 args.
    a = lang.const(2)
    self.assertEqual((a**3).eval(), 8)
    self.assertEqual((3**a).eval(), 9)
    self.assertEqual((3**a).eval(), 9)
    self.assertEqual(pow(2, 3, 5), 3)
    self.assertEqual(pow(a, 3, 5).eval(), 3)
    self.assertEqual(pow(a, a, 5).eval(), 4)
    self.assertEqual(pow(a, a, a).eval(), 0)

  def test_call_promotion(self):
    a = lang.const(lambda x, y: x + y)
    self.assertEqual(a(1, 4).eval(), 5)
    self.assertEqual(a(1, y=4).eval(), 5)
    self.assertEqual(a(x=1, y=4).eval(), 5)

if __name__ == "__main__":
  absltest.main()
