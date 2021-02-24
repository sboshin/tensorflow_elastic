# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Use ei in python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
import os
import sys
import tensorflow as tf

#tf_version = ".".join(tf.__version__.split(".")[:2])
#supported_versions = ["2.0"]
#if (tf_version in supported_versions):
tf_extra = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_tf_extra.so'))
print(tf_extra, type(tf_extra))
#print(dir(tf_extra), flush=True)
print(tf_extra.e_serialize_multi_device_iterator, flush=True)
#print(tf_extra.dataset_ops, flush=True)
#print(tf_extra.tf_extra, flush=True)
#else:
#  raise ValueError(
#      "Tensorflow Version Detected %s but supported versions are %s" %
#      (tf_version, supported_versions))
