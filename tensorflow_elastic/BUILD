licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured", "if_cuda")


config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

py_library(
    name = "elastic_collective_all_reduce_strategy",
    srcs = ([
        "python/distribute/elastic_collective_all_reduce_strategy.py",
    ]),
    srcs_version = "PY2AND3",
)

py_library(
    name = "elastic_cluster_resolver",
    srcs = ([
        "python/distribute/cluster_resolver/elastic_cluster_resolver.py",
    ]),
    deps = [
      ":monkey_patches",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "elastic_cluster_keras",
    srcs = ([
        "python/keras/elastic_callback.py",
    ]),
    srcs_version = "PY2AND3",
)

py_library(
    name = "monkey_patches",
    srcs = ([
        "python/data/ops/multi_device_iterator_ops_mp.py",
        "python/distribute/input_lib_monkey_patch.py",
        "python/eager/monkey_patch.py",
        "python/distribute/monkey_patches.py",
    ]) + glob(["python/*/__init__.py"]),
    srcs_version = "PY2AND3",
)


py_library(
    name = "elastic_mwms",
    srcs = ([
      "__init__.py",
        #"python/__init__.py",
    ]) + glob(["python/**/__init__.py"]),
    srcs_version = "PY2AND3",

    deps = [
      ":elastic_collective_all_reduce_strategy",
        ":elastic_cluster_resolver",
        ":multi_device_iterator_ops_py",
        ":monkey_patches",
        ":elastic_cluster_keras",
    ]
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

cc_binary(
    name = 'python/ops/_multi_device_iterator_ops.so',
    srcs = [
        "cc/kernels/multi_device_iterator_ops.cc",
        "cc/ops/multi_device_iterator_ops.cc",
    ],
    linkshared = 1,
    features = select({
        ":windows": ["windows_export_all_symbols"],
        "//conditions:default": [],
    }),    
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@local_config_tf//:_pywrap_tensorflow",
    ],
    copts = select({
        ":windows": ["/D__CLANG_SUPPORT_DYN_ANNOTATION__", "/DEIGEN_MPL2_ONLY", "/DEIGEN_MAX_ALIGN_BYTES=64", "/DEIGEN_HAS_TYPE_TRAITS=0", "/DTF_USE_SNAPPY", "/showIncludes", "/MD", "/O2", "/DNDEBUG", "/w", "-DWIN32_LEAN_AND_MEAN", "-DNOGDI", "/d2ReducedOptimizeHugeFunctions", "/arch:AVX", "/std:c++14", "-DTENSORFLOW_MONOLITHIC_BUILD", "/DPLATFORM_WINDOWS", "/DEIGEN_HAS_C99_MATH", "/DTENSORFLOW_USE_EIGEN_THREADPOOL", "/DEIGEN_AVOID_STL_ARRAY", "/Iexternal/gemmlowp", "/wd4018", "/wd4577", "/DNOGDI", "/UTF_COMPILE_LIBRARY"],
        "//conditions:default": ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"],
    }) + if_cuda_is_configured(["-DTENSORFLOW_USE_NVCC=1", "-DGOOGLE_CUDA=1", "-x cuda", "-nvcc_options=relaxed-constexpr", "-nvcc_options=ftz=true"]),
)

py_library(
    name = "multi_device_iterator_ops_py",
    srcs = ([
        "python/ops/multi_device_iterator_ops.py",
    ]),
    data = [
        ":python/ops/_multi_device_iterator_ops.so"
    ],
    srcs_version = "PY2AND3",
)