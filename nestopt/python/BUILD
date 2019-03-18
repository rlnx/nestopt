load("@//bazel/cython:rules.bzl", "pyx_library")
load("@//bazel/python:rules.bzl", "py_package")

pyx_library(
    name = "nestopt_py",
    srcs = glob(["nestopt/**/*.py"], exclude=["nestopt/**/*_test.py"]),
    pyx_srcs = glob([ "nestopt/**/*.pyx" ]),
    pyx_deps = glob([ "nestopt/**/*.pxd" ]) + [
        "//nestopt/core:headers",
    ],
    cc_deps = [
        "@python//:headers",
        "@python//:numpy_headers",
        "//nestopt/core:nestopt_core",
    ]
)

py_test(
    name = "test_solvers",
    main = "nestopt/solvers_test.py",
    srcs = [ "nestopt/solvers_test.py" ],
    deps = [ ":nestopt_py" ],
    imports = [ "." ],
)

py_test(
    name = "test_utils",
    main = "nestopt/utils_test.py",
    srcs = [
        "nestopt/utils.py",
        "nestopt/utils_test.py",
    ],
    imports = [ "." ],
)

py_package(
    name = "nestopt",
    srcs = [ ":nestopt_py" ],
)
