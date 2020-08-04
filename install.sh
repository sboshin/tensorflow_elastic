set -e
bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts
pip install -U artifacts/tensorflow_elastic-0.0.1-cp36-cp36m-linux_x86_64.whl
