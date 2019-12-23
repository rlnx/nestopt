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
    name = "nestopt_core_c",
    srcs = glob(["**/*.c"]),
    deps = [ ":headers_c" ],
)

cc_library(
    name = "nestopt_core",
    srcs = glob([ "**/*.cpp"], exclude=[ "**/*_test.cpp" ]),
    deps = [
        ":headers",
        ":nestopt_core_c",
    ],
    copts = [ "-std=c++17" ],
    visibility = [ "//visibility:public" ],
)

cc_test(
    name = "intervals_test",
    srcs = [ "intervals_test.cpp" ],
    copts = [ "-std=c++17" ],
    deps = [
        ":nestopt_core",
        "@gtest//:gtest",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "cubes_test",
    srcs = [ "cubes_test.cpp" ],
    copts = [ "-std=c++17" ],
    deps = [
        ":nestopt_core",
        "@gtest//:gtest",
        "@gtest//:gtest_main",
    ],
)
