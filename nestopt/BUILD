load("@//tools:python.bzl", "pyx_binaries",
                            "py_package")

py_library(
    name = "pxd_headers",
    srcs = glob([ "nestoptpy/**/*.pxd" ]),
)

pyx_binaries(
    name = "nestoptpy_native",
    pyx_srcs = glob([ "nestoptpy/**/*.pyx", ]),
    pyx_deps = [
        ":pxd_headers",
        "@//nestopt/core:headers",
    ],
    cc_deps = [
        "@python//:headers",
        "@//nestopt/core:nestopt_core",
    ]
)

py_package(
    name = "nestoptpy",
    srcs = glob(["nestoptpy/**/*.py"]),
    deps = [ ":nestoptpy_native" ],
)
