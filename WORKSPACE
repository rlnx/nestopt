workspace(name = "nestopt")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Configure Python repository (stores path to the include and lib)
load("//bazel/python:repo.bzl", "python_repo")
python_repo(name = "python")

# Google Test
http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.8.1.zip",
    sha256 = "927827c183d01734cc5cfef85e0ff3f5a92ffe6188e0d18e909c5efebf28a0c7",
    strip_prefix = "googletest-release-1.8.1",
)

# PyBind11
http_archive(
    name = "pybind11",
    url = "https://github.com/pybind/pybind11/archive/v2.4.3.zip",
    sha256 = "f1cc1e9c2836f9873aefdaf76a3280a55aae51068c759b27499a9cf34090d364",
    build_file = "//bazel/pybind11:pybind11.BUILD",
    strip_prefix = "pybind11-2.4.3"
)
