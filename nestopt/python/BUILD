load("//bazel/cython:rules.bzl", "pyx_library")
load("//bazel/python:rules.bzl", "py_package")
load("//bazel/config:copts.bzl", "NESTOPT_DEFAULT_COPTS")

cc_library(
    name = "numpy_glue",
    hdrs = glob([ "nestopt/native/**/*.hpp" ]),
    copts = NESTOPT_DEFAULT_COPTS,
    deps = [
        "//nestopt/core:headers",
        "@python//:numpy_headers",
    ],
)

pyx_library(
    name = "nestopt_py",
    srcs = glob(["nestopt/**/*.py"], exclude=["nestopt/**/*_test.py"]),
    pyx_srcs = glob([ "nestopt/**/*.pyx" ]),
    pyx_deps = glob([ "nestopt/**/*.pxd" ]) + [
        "//nestopt/core:headers",
    ],
    cc_opts = NESTOPT_DEFAULT_COPTS,
    cc_deps = [
        "@python//:headers",
        "@python//:numpy_headers",
        ":numpy_glue",
        "//nestopt/core:lib",
    ],
)

py_test(
    name = "test_nested_solver",
    main = "nestopt/solvers_test.py",
    srcs = [ "nestopt/solvers_test.py" ],
    deps = [ ":nestopt_py" ],
    imports = [ "." ],
    args = [ "TestNestedSolver" ]
)

py_test(
    name = "test_adaptive_solver",
    main = "nestopt/solvers_test.py",
    srcs = [ "nestopt/solvers_test.py" ],
    deps = [ ":nestopt_py" ],
    imports = [ "." ],
    args = [ "TestAdaptiveSolver", "TestAdaptiveTask" ]
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
