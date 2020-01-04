load("//bazel/config:copts.bzl", "NESTOPT_DEFAULT_CXX_OPTS")

cc_library(
    name = "common_headers",
    hdrs = glob([ "**/*.hpp" ]),
    visibility = [ "//visibility:public" ],
)

cc_library(
    name = "common_lib",
    srcs = glob([ "**/*.cpp"], exclude=[ "**/*_test.cpp" ]),
    copts = NESTOPT_DEFAULT_CXX_OPTS,
    deps = [ ":common_headers", ],
    visibility = [ "//visibility:public" ],
)

cc_library(
    name = "headers",
    deps = [
        ":common_headers",
        "//cpp/nestopt/direct:headers",
        "//cpp/nestopt/problems:headers",
    ],
    visibility = [ "//visibility:public" ],
)

cc_library(
    name = "nestopt_lib",
    deps = [
        ":common_lib",
        "//cpp/nestopt/direct:direct_lib",
        "//cpp/nestopt/problems:problems_lib",
    ],
    visibility = [ "//visibility:public" ],
)

cc_test(
    name = "intervals_test",
    srcs = [ "intervals_test.cpp" ],
    copts = NESTOPT_DEFAULT_CXX_OPTS,
    deps = [
        ":common_lib",
        "@gtest//:gtest",
        "@gtest//:gtest_main",
    ],
)
