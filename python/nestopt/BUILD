load("//bazel/python:rules.bzl", "py_package")

py_library(
    name = "nestopt_py_lib",
    data = ["//python/nestopt/native:nestopt.so"],
    srcs = glob(["**/*.py"], exclude=["**/*_test.py"]),
)

py_test(
    name = "test_nested_solver",
    main = "solvers_test.py",
    srcs = [ "solvers_test.py" ],
    deps = [ ":nestopt_py_lib" ],
    imports = [ ".." ],
    args = [ "TestNestedSolver" ]
)

py_test(
    name = "test_adaptive_solver",
    main = "solvers_test.py",
    srcs = [ "solvers_test.py" ],
    deps = [ ":nestopt_py_lib" ],
    imports = [ ".." ],
    args = [ "TestAdaptiveSolver", "TestAdaptiveTask" ]
)

py_test(
    name = "test_utils",
    main = "utils_test.py",
    srcs = [
        "utils.py",
        "utils_test.py",
    ],
    imports = [ ".." ],
)

py_package(
    name = "nestopt_py",
    srcs = [
        ":nestopt_py_lib",
    ],
)
