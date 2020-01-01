load("//bazel/config:copts.bzl", "NESTOPT_DEFAULT_COPTS")

cc_library(
    name = "headers",
    hdrs = glob([ "**/*.hpp" ]),
    visibility = [ "//visibility:public" ],
)

cc_library(
    name = "lib",
    srcs = glob([ "**/*.cpp"], exclude=[ "**/*_test.cpp" ]),
    copts = NESTOPT_DEFAULT_COPTS,
    deps = [
        ":headers",
        "//nestopt/core:common_headers",
    ],
    visibility = [ "//visibility:public" ],
)

cc_test(
    name = "split_cube_test",
    srcs = [ "split_cube_test.cpp" ],
    copts = NESTOPT_DEFAULT_COPTS,
    deps = [
        ":lib",
        "//nestopt/core:common_lib",
        "@gtest//:gtest",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "solver_test",
    srcs = [ "solver_test.cpp" ],
    copts = NESTOPT_DEFAULT_COPTS,
    deps = [
        ":lib",
        "//nestopt/core:common_lib",
        "//nestopt/core/problems:lib",
        "@gtest//:gtest",
        "@gtest//:gtest_main",
    ],
)
