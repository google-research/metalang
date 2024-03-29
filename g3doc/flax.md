# Flax Bindings

Flax modules are natively supported within metalang, however for ease of use and
readability, additional syntactic sugar bindings are provided.

These primarily allow one to interact with a flax module object via init and
apply more naturally.

## Datasets and RNGs

Datasets and RNGs are both just treated as iterable variables. No special
treatment is needed for datasets, but RNGs additionally provide a utility for
generating an infinite sequence of RNGs from a seed RNG.

## Initialized vs Value bound params

A major distinction that is required within metalang when using flax is whether
the params generated by a module should be treated as metalang constants or
variables. If treated as variables, the value provided by the module's init
function will be the initialization value of the param expression. If treated as
constants, the result of init will by the direct value of the expression. The
primary difference is whether the resulting metalang AST expects the params to
change with subsequent invocations.

For usages where a module's parameters are to be learned in a loop of AST
invocations (e.g. the metalang AST represents a training step for the module
params) initialization should be used and `bindings.flax.init_fn` should be
invoked with `init=True`. If the module is effectively constant (e.g. a module
that implements gradient descent with fixed learning rate) then init should be
set to false.

For example:

```py
import metalang as ml

# We expect a dataset with 2 elements per entry.
rng = ml.bindings.flax.RngVar("rng")
train_ds = ml.bindings.flax.DatasetVar("train", (2, ))
module = ml.required("module")  # A flax module to optimize.
optimizer = ml.const(SomeSgdModule(.01), "sgd_module") # A const flax module

x,y = train_ds.next()
module_params = ml.bindings.flax.init_fn(module, "module_params", init=True)(rng.next(), x)

optmizer_params = ml.bindings.flax.init_fn(optimizer, "opt_params", init=False)(rng.next(), ml.constant_fn(0.0), module_params)

loss = ml.wrap(some_loss)(ml.bindings.flax.apply_fn(module)(module_params, x), y)

# This is the result after one step of SGD.
new_params = ml.bindings.flax.apply_fn(optimizer)(optimizer_params, loss.wrt(module_params), module_params)

```

## Metalang to Flax

There is (experimental) support for turning arbitrary metalang expressions into
flax modules. `ml.bindings.flax.ExprModule(expr=some_expression)` will construct
a valid flax module that takes as arguments all unbound expressions of the
provided expression and returns the result of eval. The module params will be
all initializable, unbound arguments of the expression AST.
