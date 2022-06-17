# Bindings for improved ease of use with other python libraries.
load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")
load("@rules_python//python:defs.bzl", "py_library")

licenses(["notice"])

py_library(
    name = "types",
    srcs = ["types.py"],
)

py_library(
    name = "flax",
    srcs = ["flax.py"],
    deps = [
        ":types",
        requirement("flax:core"),
        requirement("jax"),
        "//lang",
    ],
)

py_library(
    name = "flax_test",
    srcs = ["flax_test.py"],
    deps = [
        ":flax",
        "//testing/pybase",
        requirement("flax"),
        requirement("jax"),
        "//lang",
    ],
)
