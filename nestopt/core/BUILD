
cc_library(
    name = "headers",
    hdrs = glob([ "**/*.hpp" ]),
    visibility = [ "//visibility:public" ],
)

cc_library(
    name = "nestopt_core",
    srcs = glob([ "**/*.cpp" ], exclude=[ "**/*_test.cpp" ]),
    deps = [ ":headers" ],
    visibility = [ "//visibility:public" ],
)

cc_test(
    name = "intervals_test",
    srcs = [ "core/intervals_test.cpp" ],
    deps = [
        "@gtest//:gtest",
        "@gtest//:gtest_main",
    ],
)
