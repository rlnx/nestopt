load("@//tools:python.bzl", "pyx_binaries",
                            "py_package")

filegroup(
    name = "pxd_headers",
    srcs = glob([ "python/**/*.pxd" ]),
)

filegroup(
    name = "py_sources",
    srcs = glob(["python/**/*.py"]),
)

pyx_binaries(
    name = "nestoptpy_native",
    pyx_srcs = glob([ "python/**/*.pyx", ]),
    pyx_deps = [
        ":pxd_headers",
        "@//nestopt/core:headers",
    ],
    cc_deps = [
        "@python//:headers",
        "@python//:numpy_headers",
        "@//nestopt/core:nestopt_core",
    ]
)

py_package(
    name = "nestopt",
    srcs = [
        ":py_sources",
        ":nestoptpy_native",
    ],
    root = "python",
)
