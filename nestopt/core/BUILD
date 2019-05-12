
cc_library(
    name = "headers",
    hdrs = glob([ "**/*.hpp", "**/*.h" ]),
    visibility = [ "//visibility:public" ],
)

cc_library(
    name = "nestopt_core",
    srcs = glob([ "**/*.cpp", "**/*.c"  ], exclude=[ "**/*_test.cpp" ]),
    deps = [ ":headers" ],
    visibility = [ "//visibility:public" ],
)

cc_test(
    name = "intervals_test",
    srcs = [ "intervals_test.cpp" ],
    deps = [
        ":nestopt_core",
        ":nestopt_core_c",
        "@gtest//:gtest",
        "@gtest//:gtest_main",
    ],
)
