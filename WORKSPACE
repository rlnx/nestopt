workspace(name = "nestopt")

# Configure Python repository (stores path to the include and lib)
load("@//tools:python.bzl", "python_repository")
python_repository(name = "python")

# Google Test
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.8.1.zip",
    sha256 = "927827c183d01734cc5cfef85e0ff3f5a92ffe6188e0d18e909c5efebf28a0c7",
    build_file = "@gtest//:BUILD.bazel",
)
