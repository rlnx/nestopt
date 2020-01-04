load("//bazel/config:copts.bzl", "NESTOPT_DEFAULT_CXX_OPTS",
                                 "NESTOPT_DEFAULT_C_OPTS")

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
    name = "problems_lib_c",
    srcs = glob(["**/*.c"]),
    copts = NESTOPT_DEFAULT_C_OPTS,
    deps = [ ":headers_c" ],
)

cc_library(
    name = "problems_lib",
    srcs = glob([ "**/*.cpp"], exclude=[ "**/*_test.cpp" ]),
    copts = NESTOPT_DEFAULT_CXX_OPTS,
    deps = [
        ":headers",
        ":problems_lib_c",
        "//cpp/nestopt:common_headers",
    ],
    visibility = [ "//visibility:public" ],
)

