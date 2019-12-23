load("//bazel/config:copts.bzl", "NESTOPT_DEFAULT_COPTS")

cc_library(
    name = "headers_c",
    hdrs = glob([ "**/*.h" ]),
)

cc_library(
    name = "headers",
    hdrs = glob([ "**/*.hpp" ]),
    deps = [ ":headers_c" ],
    visibility = [ "//visibility:public" ],
)

cc_library(
    name = "lib_c",
    srcs = glob(["**/*.c"]),
    deps = [ ":headers_c" ],
)

cc_library(
    name = "lib",
    srcs = glob([ "**/*.cpp"], exclude=[ "**/*_test.cpp" ]),
    copts = NESTOPT_DEFAULT_COPTS,
    deps = [
        ":headers",
        ":lib_c",
        "//nestopt/core:common_headers",
    ],
    visibility = [ "//visibility:public" ],
)

