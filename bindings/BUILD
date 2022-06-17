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
    name = "metaflax",
    srcs = ["metaflax.py"],
    deps = [
        ":types",
        requirement("flax"),
        requirement("jax"),
        "//lang",
    ],
)

py_test(
    name = "metaflax_test",
    srcs = ["metaflax_test.py"],
    deps = [
        ":metaflax",
        requirement("absl-py"),
        requirement("flax"),
        requirement("jax"),
        "//lang",
    ],
)
