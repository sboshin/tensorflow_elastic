licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured", "if_cuda")

#load("@grpc_python_dependencies//:requirements.bzl", "requirement")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_github_grpc_grpc//bazel:grpc_build_system.bzl", "grpc_proto_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library", "py_proto_library")

grpc_proto_library(
  name = "orchestrator",
  srcs = ["rendezvous/proto/orchestrator.proto"]
)

proto_library(
    name = "rendezvous/proto/orchestrator_proto",
    srcs = ["rendezvous/proto/orchestrator.proto"],
)

py_proto_library(
    name = "orchestrator_py_pb2",
    deps = [":rendezvous/proto/orchestrator_proto"],
)

py_grpc_library(
    name = "orchestrator_py_pb2_grpc",
    srcs = [":rendezvous/proto/orchestrator_proto"],
    deps = [":orchestrator_py_pb2"],
)

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

py_library(
    name = "elastic_mwms",
    srcs = glob(["**/__init__.py"])+ ["version.py"],
    srcs_version = "PY2AND3",

    deps= [
      ":tensorflow_elastic_impl",
      ":orchestrator_py_pb2_grpc",
    ]

)

py_library(
    name = "tensorflow_elastic_impl",
    srcs = [
      "__init__.py",
        #"python/__init__.py",
    ] + glob(["agent/**/*.py"])
    + glob(["distributed/**/*.py"])
    + glob(["events/**/*.py"])
    + glob(["metrics/**/*.py"])
    + glob(["rendezvous/**/*.py"])
    + glob(["timer/**/*.py"])
    + glob(["utils/**/*.py"])
    + glob(["tensorflow/**/*.py"])
    + glob(["api/**/*.py"]),
    srcs_version = "PY2AND3",

    deps = [":tf_extra_py"],

)

py_library(
  name = "testing_utils",
  srcs = glob(["tests/bin/*.py", "tests/utils.py"])
)

py_test(
    name = "multi_device_iterator_test",
    srcs = [
        "tensorflow/multi_device_iterator_test.py",
    ],
    deps = [
        "elastic_mwms",
        ":testing_utils",
    ],
    srcs_version = "PY2AND3",
    timeout = "eternal",
)

py_test(
    name = "cluster_resolver_test",
    srcs = [
        "tests/cluster_resolver_test.py",
    ] + glob(["tests/bin/*.py"]),
    deps = [
        "elastic_mwms",
    ],
    srcs_version = "PY2AND3",
    timeout = "eternal",
)

py_test(
    name = "orchestrator_server_test",
    srcs = [
        "rendezvous/orchestrator_server_test.py",
    ],
    deps = [
        "elastic_mwms",
        ":orchestrator_py_pb2_grpc",
    ],
    srcs_version = "PY2AND3",
    timeout = "long",
)

py_test(
    name = "worker_restart_test",
    srcs = [
        "tests/worker_restart_test.py",
    ],
    deps = [
        "elastic_mwms",
        ":orchestrator_py_pb2_grpc",
    ],
    srcs_version = "PY2AND3",
    timeout = "long",
)

py_test(
    name = "worker_norestart_test",
    srcs = [
        "tests/worker_norestart_test.py",
    ],
    deps = [
        "elastic_mwms",
        ":orchestrator_py_pb2_grpc",
        ":testing_utils",
    ],
    srcs_version = "PY2AND3",
    timeout = "long",
)

#cc_library(
#  name = "dataset_deps",
#  hdrs = ["tensorflow/kernels/dataset_utils.cc"],
#  srcs = ["tensorflow/kernels/dataset_utils.cc"],
#  deps = ["@local_config_tf//:tf_header_lib"]
#)

cc_binary(
    name = 'tensorflow/python/_tf_extra.so',
    srcs = [
            "tensorflow/kernels/multi_device_iterator_ops.cc", 
            "tensorflow/kernels/unbounded_threadpool.cc",
            #"tensorflow/kernels/dataset_utils.cc"
            ]
            + glob(["tensorflow/ops/*.cc"]),
    
    linkshared = 1,
    includes = ["tensorflow/kernels"],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:pywrap_framework",
        "@local_config_tf//:tf_header_lib",
        #":dataset_deps",
    ],
    copts = select({
        "//conditions:default": ["-pthread", "-std=c++14", "-D_GLIBCXX_USE_CXX11_ABI=0"],
    }) + if_cuda_is_configured(["-DTENSORFLOW_USE_NVCC=1", "-DGOOGLE_CUDA=1", "-x cuda", "-nvcc_options=relaxed-constexpr", "-nvcc_options=ftz=true"]),
)

py_library(
    name = "tf_extra_py",
    srcs = ([
        "tensorflow/python/tf_extra.py",
    ]),
    #data = [
    #    ":tensorflow/python/_tf_extra.so"
    #],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)