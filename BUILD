# Meta-language for lazy evaluation of Flax modules.

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

licenses(["notice"])

exports_files(["LICENSE"])

pytype_strict_library(
    name = "metalang",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/metalang/lang:consts",
        "//third_party/py/metalang/lang:expr",
        "//third_party/py/metalang/lang:functions",
    ],
)
