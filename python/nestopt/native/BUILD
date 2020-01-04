load("//bazel/config:copts.bzl", "NESTOPT_DEFAULT_CXX_OPTS")

cc_library(
    name = "headers",
    hdrs = glob(["*.hpp"]),
    copts = NESTOPT_DEFAULT_CXX_OPTS,
    deps = [
        "@pybind11//:headers",
        "//cpp/nestopt:headers",
    ],
)

cc_binary(
    name = "nestopt.so",
    srcs = ["nestopt.cpp"],
    deps = [
        ":headers",
        "//cpp/nestopt:nestopt_lib",
    ],
    copts = NESTOPT_DEFAULT_CXX_OPTS,
    linkopts = [ '-Wl,-undefined,dynamic_lookup' ],
    linkshared = 1,
    visibility = [ "//visibility:public" ],
)
