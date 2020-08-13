load("//tf:tf_configure.bzl", "tf_configure")
load("//gpu:cuda_configure.bzl", "cuda_configure")

tf_configure(name = "local_config_tf")

cuda_configure(name = "local_config_cuda")
load(":custom.bzl", "clean_dep", "ei_http_archive", "bind_custom")

#ei_http_archive(
#        name = "grpc",
#        sha256 = "67a6c26db56f345f7cee846e681db2c23f919eba46dd639b09462d1b6203d28c",
#        strip_prefix = "grpc-4566c2a29ebec0835643b972eb99f4306c4234a3",
#        system_build_file = clean_dep("//third_party/grpc.BUILD"),
#        urls = [
#            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/grpc/grpc/archive/4566c2a29ebec0835643b972eb99f4306c4234a3.tar.gz",
#            "https://github.com/grpc/grpc/archive/4566c2a29ebec0835643b972eb99f4306c4234a3.tar.gz",
#        ],
#    )

ei_http_archive(
        name = "com_github_grpc_grpc",
        sha256 = "b956598d8cbe168b5ee717b5dafa56563eb5201a947856a6688bbeac9cac4e1f",
        strip_prefix = "grpc-b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd",
        system_build_file = clean_dep("//third_party/systemlibs:grpc.BUILD"),
        system_link_files = {
            "//third_party/systemlibs:BUILD": "bazel/BUILD",
            "//third_party/systemlibs:grpc.BUILD": "src/compiler/BUILD",
            #"//third_party/systemlibs:grpc.bazel.grpc_deps.bzl": "bazel/grpc_deps.bzl",
        },
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/grpc/grpc/archive/b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd.tar.gz",
            "https://github.com/grpc/grpc/archive/b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd.tar.gz",
        ],
    )

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load(
    "@build_bazel_rules_apple//apple:repositories.bzl",
    "apple_rules_dependencies",
)

apple_rules_dependencies()

load(
    "@build_bazel_apple_support//lib:repositories.bzl",
    "apple_support_dependencies",
)

apple_support_dependencies()

load("@upb//bazel:repository_defs.bzl", "bazel_version_repository")

bazel_version_repository(name = "bazel_version")

#load("//third_party/googleapis:repository_rules.bzl", "config_googleapis")

#config_googleapis()