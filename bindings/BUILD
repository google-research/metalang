# Bindings for improved ease of use with other python libraries.
load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

licenses(["notice"])

pytype_strict_library(
    name = "types",
    srcs = ["types.py"],
)

pytype_strict_library(
    name = "flax",
    srcs = ["flax.py"],
    deps = [
        ":types",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/metalang/lang",
    ],
)

py_strict_test(
    name = "flax_test",
    srcs = ["flax_test.py"],
    deps = [
        ":flax",
        "//testing/pybase",
        "//third_party/py/flax",
        "//third_party/py/jax",
        "//third_party/py/metalang/lang",
    ],
)
