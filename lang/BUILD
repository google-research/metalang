# Meta-language for lazy evaluation of Flax modules.

load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

licenses(["notice"])

pytype_strict_library(
    name = "functions",
    srcs = ["functions.py"],
    visibility = ["//third_party/py/metalang:__subpackages__"],
    deps = [
        ":expr",
    ],
)

pytype_strict_library(
    name = "lang",
    srcs = ["__init__.py"],
    visibility = ["//third_party/py/metalang:__subpackages__"],
    deps = [
        ":consts",
        ":expr",
        ":functions",
    ],
)

pytype_strict_library(
    name = "expr",
    srcs = ["expr.py"],
    visibility = ["//third_party/py/metalang:__subpackages__"],
    deps = [
        "//third_party/py/chex",
        "//third_party/py/immutabledict",
        "//third_party/py/jax",
    ],
)

pytype_strict_library(
    name = "consts",
    srcs = ["consts.py"],
    visibility = ["//third_party/py/metalang:__subpackages__"],
    deps = [":expr"],
)

py_strict_test(
    name = "lang_test",
    srcs = ["lang_test.py"],
    deps = [
        ":lang",
        "//testing/pybase",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)
