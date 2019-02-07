cc_library(
    name = "headers",
    hdrs = glob([ "**/*.hpp" ]),
    visibility = [ "//visibility:public" ],
)

cc_library(
    name = "nestopt_core",
    srcs = glob([ "**/*.cpp" ]),
    deps = [ ":headers" ],
    visibility = [ "//visibility:public" ],
)
