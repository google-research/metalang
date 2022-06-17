# Meta-language for lazy evaluation of Flax modules.

load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")
load("@rules_python//python:defs.bzl", "py_library")

licenses(["notice"])

py_library(
    name = "functions",
    srcs = ["functions.py"],
    visibility = ["//third_party/py/metalang:__subpackages__"],
    deps = [
        ":expr",
    ],
)

py_library(
    name = "lang",
    srcs = ["__init__.py"],
    visibility = ["//third_party/py/metalang:__subpackages__"],
    deps = [
        ":consts",
        ":expr",
        ":functions",
    ],
)

py_library(
    name = "expr",
    srcs = ["expr.py"],
    visibility = ["//third_party/py/metalang:__subpackages__"],
    deps = [
        requirement("chex"),
        requirement("immutabledict"),
        requirement("jax"),
    ],
)

py_library(
    name = "consts",
    srcs = ["consts.py"],
    visibility = ["//third_party/py/metalang:__subpackages__"],
    deps = [":expr"],
)

py_library(
    name = "lang_test",
    srcs = ["lang_test.py"],
    deps = [
        ":lang",
        "//testing/pybase",
        requirement("jax"),
        requirement("numpy"),
    ],
)
