licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured", "if_cuda")


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
    + glob(["utils/**/*.py"]),
    srcs_version = "PY2AND3",

)


py_test(
    name = "elastic_collective_all_reduce_strategy_test",
    srcs = [
        "python/distribute/elastic_collective_all_reduce_strategy_test.py",
    ],
    deps = [
        "elastic_mwms",
    ],
    srcs_version = "PY2AND3",
)
