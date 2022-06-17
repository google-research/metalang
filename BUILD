# Meta-language for lazy evaluation of Flax modules.

load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library")

licenses(["notice"])

exports_files([
    "LICENSE",
    "requirements.txt",
])

py_library(
    name = "metalang",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//lang:consts",
        "//lang:expr",
        "//lang:functions",
    ],
)
