load("@//third_party:cython.bzl", "pyx_library")

cc_library(
    name = "core_lib",
    hdrs = glob(["core/**/*.hpp"]),
    srcs = glob(["core/**/*.cpp"]),
    visibility= [ "//visibility:public" ],
)

cc_library(
    name = "nestopt_lib",
    deps = [ ":core_lib" ],
)

py_library(
    name = "nestopt_py",
    srcs = glob(["python/**/*.py"]),
)

pyx_library(
    name = "nestopt_native",
    srcs = glob([
        "python/native/**/*.pyx",
        "python/native/**/*.pxd"
    ]),
    deps = [
        ":nestopt_lib",
        "@python//:headers",
    ],
    py_deps = [ ":nestopt_py" ],
)
