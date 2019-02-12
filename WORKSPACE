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
    strip_prefix = "googletest-release-1.8.1",
)

# http_archive(
#      name = "google",
#      urls = ["https://github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip"],  # 2019-01-07
#      strip_prefix = "googletest-b6cd405286ed8635ece71c72f118e659f4ade3fb",
#      sha256 = "ff7a82736e158c077e76188232eac77913a15dac0b22508c390ab3f88e6d6d86",
# )
