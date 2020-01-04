load("//bazel/config:copts.bzl", "NESTOPT_DEFAULT_CXX_OPTS")

cc_library(
    name = "headers",
    hdrs = glob([ "**/*.hpp" ]),
    visibility = [ "//visibility:public" ],
)

cc_library(
    name = "direct_lib",
    srcs = glob([ "**/*.cpp"], exclude=[ "**/*_test.cpp" ]),
    copts = NESTOPT_DEFAULT_CXX_OPTS,
    deps = [
        ":headers",
        "//cpp/nestopt:common_headers",
    ],
    visibility = [ "//visibility:public" ],
)

cc_test(
    name = "split_cube_test",
    srcs = [ "split_cube_test.cpp" ],
    copts = NESTOPT_DEFAULT_CXX_OPTS,
    deps = [
        ":direct_lib",
        "//cpp/nestopt:common_lib",
        "@gtest//:gtest",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "solver_test",
    srcs = [ "solver_test.cpp" ],
    copts = NESTOPT_DEFAULT_CXX_OPTS,
    deps = [
        ":direct_lib",
        "//cpp/nestopt:common_lib",
        "//cpp/nestopt/problems:problems_lib",
        "@gtest//:gtest",
        "@gtest//:gtest_main",
    ],
)
