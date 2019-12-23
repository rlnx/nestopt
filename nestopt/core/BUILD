load("//bazel/config:copts.bzl", "NESTOPT_DEFAULT_COPTS")

cc_library(
    name = "common_headers_c",
    hdrs = glob([ "**/*.h" ]),
)

cc_library(
    name = "common_lib_c",
    srcs = glob(["**/*.c"]),
    deps = [ ":common_headers_c" ],
)

cc_library(
    name = "common_headers",
    hdrs = glob([ "**/*.hpp" ]),
    deps = [ ":common_headers_c" ],
    visibility = [ "//visibility:public" ],
)

cc_library(
    name = "common_lib",
    srcs = glob([ "**/*.cpp"], exclude=[ "**/*_test.cpp" ]),
    copts = NESTOPT_DEFAULT_COPTS,
    deps = [
        ":common_headers",
        ":common_lib_c",
    ],
    visibility = [ "//visibility:public" ],
)

cc_library(
    name = "headers",
    deps = [
        ":common_headers",
        "//nestopt/core/direct:headers",
    ],
    visibility = [ "//visibility:public" ],
)

cc_library(
    name = "lib",
    deps = [
        ":common_lib",
        "//nestopt/core/direct:lib"
    ],
    visibility = [ "//visibility:public" ],
)

cc_test(
    name = "intervals_test",
    srcs = [ "intervals_test.cpp" ],
    copts = NESTOPT_DEFAULT_COPTS,
    deps = [
        ":common_lib",
        "@gtest//:gtest",
        "@gtest//:gtest_main",
    ],
)
