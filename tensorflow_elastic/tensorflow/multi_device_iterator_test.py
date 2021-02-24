# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for `tf.data.MultiDeviceIterator`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import time
import multiprocessing as mp

from tensorflow_elastic import tensorflow
from tensorflow_elastic.tests import utils
import numpy as np
from absl.testing import parameterized
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import iterator_ops as contrib_iterator_ops
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceStrategy


def skip_v2_test_combinations():
  # TODO(b/121264236): Support v2 behavior for these tests.
  return combinations.combine(tf_api_version=1, mode=["eager", "graph"])


class MultiDeviceIteratorTest(test_base.DatasetTestBase,
                              parameterized.TestCase):

  @combinations.generate(
      combinations.times(skip_v2_test_combinations(),
                         combinations.combine(num_inits=[0, 1, 42])))
  def testInitOnly(self, num_inits):
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"])

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config):
      for _ in range(num_inits):
        self.evaluate(multi_device_iterator.initializer)

  @combinations.generate(skip_v2_test_combinations())
  def testBasic(self):
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"])

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config):
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 10, 2):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.assertEqual(i, self.evaluate(elem_on_1))
        self.assertEqual(i + 1, self.evaluate(elem_on_2))
      with self.assertRaises(errors.OutOfRangeError):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)

  @combinations.generate(skip_v2_test_combinations())
  def testOneOnSameDevice(self):
    with ops.device("/cpu:0"):
      dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:0", "/cpu:1"])

    config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=config):
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 10, 2):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.assertEqual(i, self.evaluate(elem_on_1))
        self.assertEqual(i + 1, self.evaluate(elem_on_2))
      with self.assertRaises(errors.OutOfRangeError):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)

  @combinations.generate(skip_v2_test_combinations())
  def testRepeatDevices(self):
    with ops.device("/cpu:0"):
      dataset = dataset_ops.Dataset.range(20)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2", "/cpu:1", "/cpu:2"])

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config):
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 20, 4):
        elements = multi_device_iterator.get_next()
        elem_on_1, elem_on_2, elem_on_3, elem_on_4 = elements
        self.assertEqual(i, self.evaluate(elem_on_1))
        self.assertEqual(i + 1, self.evaluate(elem_on_2))
        self.assertEqual(i + 2, self.evaluate(elem_on_3))
        self.assertEqual(i + 3, self.evaluate(elem_on_4))
      with self.assertRaises(errors.OutOfRangeError):
        elements = multi_device_iterator.get_next()
        elem_on_1, elem_on_2, elem_on_3, elem_on_4 = elements
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)
        self.evaluate(elem_on_3)
        self.evaluate(elem_on_4)

  @combinations.generate(skip_v2_test_combinations())
  def testNotFullyDivisible(self):
    dataset = dataset_ops.Dataset.range(9)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"])

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config):
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 8, 2):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.assertEqual(i, self.evaluate(elem_on_1))
        self.assertEqual(i + 1, self.evaluate(elem_on_2))
      elem_on_1 = multi_device_iterator.get_next("/cpu:1")
      self.assertEqual(8, self.evaluate(elem_on_1))
      with self.assertRaises(errors.OutOfRangeError):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)

  @combinations.generate(skip_v2_test_combinations())
  def testGetNextAsOptional(self):
    if context.executing_eagerly():
      return

    dataset = dataset_ops.Dataset.range(9)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"])
    elem_on_1, elem_on_2 = multi_device_iterator.get_next_as_optional()
    elem_on_1_has_value_t = elem_on_1.has_value()
    elem_on_1_t = elem_on_1.get_value()
    elem_on_2_has_value_t = elem_on_2.has_value()
    elem_on_2_t = elem_on_2.get_value()

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config) as sess:
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 8, 2):
        elem_on_1_has_value, elem_on_1_value = sess.run(
            [elem_on_1_has_value_t, elem_on_1_t])
        self.assertTrue(elem_on_1_has_value)
        self.assertEqual(i, elem_on_1_value)
        elem_on_2_has_value, elem_on_2_value = sess.run(
            [elem_on_2_has_value_t, elem_on_2_t])
        self.assertTrue(elem_on_2_has_value)
        self.assertEqual(i + 1, elem_on_2_value)
      elem_on_1_has_value, elem_on_1_value = sess.run(
          [elem_on_1_has_value_t, elem_on_1_t])
      self.assertTrue(elem_on_1_has_value)
      self.assertEqual(8, elem_on_1_value)
      self.assertFalse(self.evaluate(elem_on_1_has_value_t))
      self.assertFalse(self.evaluate(elem_on_2_has_value_t))
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(elem_on_1_t)
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(elem_on_2_t)

  @combinations.generate(skip_v2_test_combinations())
  def testUneven(self):
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"], max_buffer_size=4)

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config):
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 10, 2):
        elem_on_1 = multi_device_iterator.get_next("/cpu:1")
        self.assertEqual(i, self.evaluate(elem_on_1))
      for i in range(0, 10, 2):
        elem_on_2 = multi_device_iterator.get_next("/cpu:2")
        self.assertEqual(i + 1, self.evaluate(elem_on_2))
      with self.assertRaises(errors.OutOfRangeError):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)

  @combinations.generate(skip_v2_test_combinations())
  def testMultipleInitializationsGraph(self):
    if context.executing_eagerly():
      return

    with ops.device("/cpu:0"):
      epoch = array_ops.placeholder(dtypes.int64, shape=[])
      dataset1 = dataset_ops.Dataset.from_tensors(epoch).repeat(1000)
      dataset2 = dataset_ops.Dataset.range(1000)
      dataset = dataset_ops.Dataset.zip((dataset1, dataset2))
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"], prefetch_buffer_size=4)
    elem_on_1, elem_on_2 = multi_device_iterator.get_next()
    init_op = multi_device_iterator.initializer

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    pool = config.session_inter_op_thread_pool.add()
    pool.num_threads = 2
    with session.Session(config=config) as sess:
      for i in range(1000):
        sess.run(init_op, feed_dict={epoch: i})
        self.assertEqual([(i, 0), (i, 1)], self.evaluate([elem_on_1,
                                                          elem_on_2]))

  @combinations.generate(skip_v2_test_combinations())
  def testMultipleInitializationsEager(self):
    if not context.executing_eagerly():
      return

    with ops.device("/cpu:0"):
      dataset1 = dataset_ops.Dataset.range(1000)
      dataset2 = dataset_ops.Dataset.range(1000)
      dataset = dataset_ops.Dataset.zip((dataset1, dataset2))

    for _ in range(5):
      multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
          dataset, ["/cpu:1", "/cpu:2"], prefetch_buffer_size=4)
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.assertEqual([(0, 0), (1, 1)], self.evaluate([elem_on_1, elem_on_2]))

  @combinations.generate(skip_v2_test_combinations())
  def testBasicGpu(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/gpu:0"])

    config = config_pb2.ConfigProto(device_count={"CPU": 2, "GPU": 1})
    with self.test_session(config=config):
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 10, 2):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.assertEqual(i, self.evaluate(elem_on_1))
        self.assertEqual(i + 1, self.evaluate(elem_on_2))
      with self.assertRaises(errors.OutOfRangeError):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)

  @combinations.generate(skip_v2_test_combinations())
  def testUnevenGpu(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/gpu:0"], max_buffer_size=4)

    config = config_pb2.ConfigProto(device_count={"CPU": 2, "GPU": 1})
    with self.test_session(config=config):
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 10, 2):
        elem_on_1 = multi_device_iterator.get_next("/cpu:1")
        self.assertEqual(i, self.evaluate(elem_on_1))
      for i in range(0, 10, 2):
        elem_on_2 = multi_device_iterator.get_next("/gpu:0")
        self.assertEqual(i + 1, self.evaluate(elem_on_2))
      with self.assertRaises(errors.OutOfRangeError):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)

  @combinations.generate(skip_v2_test_combinations())
  def testGetNextAsOptionalGpu(self):
    if not test_util.is_gpu_available() or context.executing_eagerly():
      self.skipTest("No GPU available")

    dataset = dataset_ops.Dataset.range(9)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/gpu:0"])
    elem_on_1, elem_on_2 = multi_device_iterator.get_next_as_optional()
    elem_on_1_has_value_t = elem_on_1.has_value()
    elem_on_1_t = elem_on_1.get_value()
    elem_on_2_has_value_t = elem_on_2.has_value()
    elem_on_2_t = elem_on_2.get_value()

    config = config_pb2.ConfigProto(device_count={"CPU": 2, "GPU": 1})
    with self.test_session(config=config) as sess:
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 8, 2):
        elem_on_1_has_value, elem_on_1_value = sess.run(
            [elem_on_1_has_value_t, elem_on_1_t])
        self.assertTrue(elem_on_1_has_value)
        self.assertEqual(i, elem_on_1_value)
        elem_on_2_has_value, elem_on_2_value = sess.run(
            [elem_on_2_has_value_t, elem_on_2_t])
        self.assertTrue(elem_on_2_has_value)
        self.assertEqual(i + 1, elem_on_2_value)
      elem_on_1_has_value, elem_on_1_value = sess.run(
          [elem_on_1_has_value_t, elem_on_1_t])
      self.assertTrue(elem_on_1_has_value)
      self.assertEqual(8, elem_on_1_value)
      self.assertFalse(self.evaluate(elem_on_1_has_value_t))
      self.assertFalse(self.evaluate(elem_on_2_has_value_t))
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(elem_on_1_t)
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(elem_on_2_t)

  @combinations.generate(skip_v2_test_combinations())
  def testOptimization(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(testing.assert_next(["MemoryCacheImpl"]))
    dataset = dataset.skip(0)  # this should be optimized away
    dataset = dataset.cache()

    options = dataset_ops.Options()
    options.experimental_optimization.noop_elimination = True
    dataset = dataset.with_options(options)

    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"])

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config):
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 10, 2):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.assertEqual(i, self.evaluate(elem_on_1))
        self.assertEqual(i + 1, self.evaluate(elem_on_2))
      with self.assertRaises(errors.OutOfRangeError):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)

  def _ckpt_path(self):
    return os.path.join(self.get_temp_dir(), "iterator")

  def _latest_ckpt(self):
    return checkpoint_management.latest_checkpoint(self.get_temp_dir())

  def _save(self, sess, saver):
    saver.save(sess, self._ckpt_path())

  def _restore(self, saver, sess):
    sess.run(lookup_ops.tables_initializer())
    saver.restore(sess, self._latest_ckpt())

  @combinations.generate(combinations.combine(mode=["graph"]))
  def testSaveIterator(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(testing.assert_next(["MemoryCacheImpl"]))
    dataset = dataset.skip(0)  # this should be optimized away
    dataset = dataset.cache()
    print(self.get_temp_dir(), flush=True)
    options = dataset_ops.Options()
    options.experimental_optimization.noop_elimination = True
    dataset = dataset.with_options(options)
    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config) as sess:
      multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
          dataset, ["/cpu:1", "/cpu:2"], prefetch_buffer_size=0)
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 5, 2):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        elem_on_1, elem_on_2 = (self.evaluate(elem_on_1),
                                self.evaluate(elem_on_2))
        print(elem_on_1, elem_on_2, flush=True)
        self.assertEqual(i, elem_on_1)
        self.assertEqual(i + 1, elem_on_2)
      #Save
      ops.add_to_collection(
          ops.GraphKeys.SAVEABLE_OBJECTS,
          multi_device_iterator_ops.make_saveable_from_iterator(
              multi_device_iterator))
      saver = saver_lib.Saver(allow_empty=True)
      self._save(sess, saver)

    print(os.listdir(self.get_temp_dir()), flush=True)
    print("\n\n\n\n\n\n\n\n\n\n\n", flush=True)

    with self.test_session(config=config) as sess:
      self._restore(saver, sess)
      for i in range(6, 10, 2):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        elem_on_1, elem_on_2 = (self.evaluate(elem_on_1),
                                self.evaluate(elem_on_2))
        print(elem_on_1, elem_on_2, flush=True)
        self.assertEqual(i, elem_on_1)
        self.assertEqual(i + 1, elem_on_2)

      with self.assertRaises(errors.OutOfRangeError):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)


class ElasticIteratorTest(utils.MultiWorkerTestBase, parameterized.TestCase):

  def _run_distribute_dataset_set(self, index, **kwargs):
    cluster_spec = {
        "cluster": {
            "worker": []
        },
        "task": {
            "index": index,
            "type": "worker"
        }
    }
    num_workers = 3
    cluster = [f"localhost:{5000+ii}" for ii in range(num_workers)]
    #cluster_spec["task"]["index"] = 0
    cluster_spec["cluster"]["worker"] = cluster
    os.environ["TF_CONFIG"] = json.dumps(cluster_spec)
    print(os.environ["TF_CONFIG"], flush=True)
    print("\n\n\n\n\n\n\n")
    strategy = CollectiveAllReduceStrategy()
    print("Here1")
    
    dataset = dataset_ops.Dataset.from_tensor_slices(
        [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]).batch(3)
    dataset = strategy.experimental_distribute_dataset(dataset)
    
    it = strategy.extended.get_elastic_iterator(dataset)
    
    for ii in range(1, 3):
      val = next(it)
      assert ii == val
      print(f"{index}|{val}")
    
    self.Barrier(kwargs["proc_id"])
    #reshard
    print("Barrier done")
    #Update cluster spec
    cluster = cluster[:-1]
    cluster_spec["cluster"]["worker"] = cluster
    os.environ["TF_CONFIG"] = json.dumps(cluster_spec)
    print(os.environ["TF_CONFIG"], flush=True)

    strategy.extended.update_cluster()
    for ii in range(3, 7):
      val = next(it)
      print(f"{index}|{val}", flush=True)
      assert ii == val
      

  def test_distribute_shrink(self):
    num_workers = 3
    self._create_barrier_array(num_workers)
    for ii in range(num_workers):
      self._start_node(self._run_distribute_dataset_set, params={"index": ii})
    self.ClientBarrier(num_workers)
    self._end_node(2)
    #time.sleep(20)
    self.wait.set()
    self.join_all()


if __name__ == "__main__":
  ops.enable_eager_execution(config=config_pb2.ConfigProto(device_count={
      "CPU": 3,
      "GPU": 1
  }))
  test.main()
