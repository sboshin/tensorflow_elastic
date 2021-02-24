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
"""Tests for CollectiveAllReduceStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import os
#os.environ["TF_CPP_VMODULE"] ="collective_param_resolver_distributed=2,collective_param_resolver_local=2,collective_ops=3,base_collective_executor=3,hierarchical_tree_broadcaster=3"
#os.environ["TF_CPP_MIN_VLOG_LEVEL"] ="0"
import json
import shutil
import psutil
import signal

#os.environ["TF_CPP_VMODULE"] ="collective_param_resolver_distributed=2,collective_param_resolver_local=2,collective_ops=3,base_collective_executor=3,hierarchical_tree_broadcaster=3"
#os.environ["TF_CPP_MIN_VLOG_LEVEL"] ="0"
from multiprocessing import Process, Queue, set_start_method
import subprocess
import time
from tensorflow.python.platform import test
from tensorflow_elastic.distributed.launch import main
from tensorflow_elastic.rendezvous.orchestrator_server import serve

import tensorflow_elastic.rendezvous.orchestrator_api as rdzv
import tensorflow_elastic.tensorflow
from tensorflow.python.distribute import collective_all_reduce_strategy
import tensorflow_elastic.tensorflow


SERVER_ADDRESS = 'localhost:50051'

def path(script):
    return os.path.join(os.path.dirname(__file__), script)

def kill_proc_tree(pid, sig=signal.SIGTERM, include_parent=True,
                   timeout=None, on_terminate=None):
    """Kill a process tree (including grandchildren) with signal
    "sig" and return a (gone, still_alive) tuple.
    "on_terminate", if specified, is a callabck function which is
    called as soon as a child terminates.
    """
    assert pid != os.getpid(), "won't kill myself"
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    if include_parent:
        children.append(parent)
    for p in children[::-1]:
        print(f"{p} Process {p.cmdline()} : {p}", flush=True)
        p.send_signal(sig)
        p.kill()
        p.terminate()
    gone, alive = psutil.wait_procs(children, timeout=timeout,
                                    callback=on_terminate)
    print(f"Gone {gone} Alive {alive}", flush=True)
    return (gone, alive)

class ElasticParams(object):
  def __init__(self, address, nnodes="1:3"):
    self.nnodes=nnodes
    nodes = nnodes.split(":")
    self.min_nodes = int(nodes[0])
    self.max_nodes = int(nodes[1])
    self.address = address

    

class ElasticTensorflowWorkerRestart(test.TestCase):
  def setUp(self):
    # Setup the grpc server
    args = [
            f"--standalone",
            f"--rdzv_endpoint={SERVER_ADDRESS}",
           "ls"]
    self._t = Process(target=main, args=(args,))
    self._t.start()
    time.sleep(2)
    self._procs = []
    self._exclude_procs = []
  
  def tearDown(self):
    print("Terminating", flush=True)
    self._t.terminate()
    self._t.join()
    
  def join_all(self):
    for ii, proc in enumerate(self._procs):
      self._procs[ii].join()
      print(self._procs[ii])
      if(self._procs[ii].exitcode != 0 and ii not in self._exclude_procs):
        raise RuntimeError("Process %d failed with exitcode %d but wasn't in exclude list"%(ii, self._procs[ii].exitcode))

  def _run_in_launch(self, params):
    # ['/home/ubuntu/anaconda3/envs/tensorflow2_p36/lib/python3.6/site-packages/tensorflow_elastic/distributed/launch.py', 
    # '--nnodes=1:2', '--nproc_per_node=1', '--rdzv_id=5', '--rdzv_backend=etcd', '--rdzv_endpoint=localhost:5379', 'ls']
    args = [
            f"--nnodes={params.nnodes}",
            f"--rdzv_endpoint={SERVER_ADDRESS}",
            f"--monitor_interval=1",
            f"--start_method=fork",
            #f"--no_python",
        ]

    script_args = [path(params.script)]+ params.args
    print("These are the args "," ".join(args+script_args))
    main(args+script_args)

  def _create_cluster_spec(self, worker_list, worker_index):
    cluser_spec_template = {"cluster":{"worker":worker_list}, "task":{"index": worker_index, "type":"worker"}}
    ret = json.dumps(cluser_spec_template)
    return ret

  def _start_node(self, target, params):
    #Use launc to start the process
    
    t = Process(target=target, kwargs=params)
    print("Starting ",str(target))
    t.start()
    self._procs.append(t)

  def _end_node(self, node_num):
    #Use launc to start the process
    print(f"Terminating process {node_num}", flush=True)
    proc = self._procs.pop(node_num)
    kill_proc_tree(proc.pid, timeout=2, include_parent=True)
    #time.sleep(1)
    proc.terminate()
    proc.join()
    print(f"After terminate, {proc.is_alive()}", flush=True)
    
  def _run_vanilla_tfconfig_test(self, tfconfig, sleep=15):
    import tensorflow as tf
    os.environ["TF_CONFIG"] = tfconfig
    strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()
    tensor_input = tf.constant(1.0)

    def ret_val(t_in):
      return t_in
    var_val = strategy.run(ret_val, args=(tensor_input,))
    val = strategy.reduce(tf.distribute.ReduceOp.SUM, var_val, None)
    print(val, flush=True)
    time.sleep(sleep)
    val = strategy.reduce(tf.distribute.ReduceOp.SUM, var_val, None)
    print(val, flush=True)
    
  def _run_elastic_tfconfig_test(self, params, sleep=15):
    import tensorflow as tf
    
    handler = rdzv.TFEOrchestratorHandler(SERVER_ADDRESS, params.min_nodes, params.max_nodes)
    prev_tfconfig = handler.GetClusterSpec(params.address, False)
    prev_tfconfig["cluster"]["worker"].sort()
    prev_tfconfig["task"]["index"] = prev_tfconfig["cluster"]["worker"].index(params.address)
    os.environ["TF_CONFIG"] = json.dumps(prev_tfconfig)
    prev_tfconfig = os.environ["TF_CONFIG"]
    print(prev_tfconfig, flush=True)
    
    strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()
    tensor_input = tf.constant(1.0)

    def ret_val(t_in):
      return t_in
    var_val = strategy.run(ret_val, args=(tensor_input,))
    val = strategy.reduce(tf.distribute.ReduceOp.SUM, var_val, None)
    print(val, flush=True)
    time.sleep(sleep)

    tfconfig = handler.GetClusterSpec(params.address, False)
    tfconfig["cluster"]["worker"].sort()
    tfconfig["task"]["index"] = tfconfig["cluster"]["worker"].index(params.address)
    
    os.environ["TF_CONFIG"] = json.dumps(tfconfig)
    assert prev_tfconfig != os.environ["TF_CONFIG"], (prev_tfconfig, os.environ["TF_CONFIG"])
    print(os.environ["TF_CONFIG"], flush=True)
    time.sleep(5)
    strategy.extended.update_cluster()
    #strategy.configure(tfconfig)
    val = strategy.reduce(tf.distribute.ReduceOp.SUM, var_val, None)
    print(val, flush=True)

  def test_tfconfig(self):
    w_list = [f"localhost:555{ii}" for ii in range(2)]
    for ii in range(2):

      self._start_node(self._run_vanilla_tfconfig_test, {"tfconfig":self._create_cluster_spec(w_list, ii), "sleep":3})
    self.join_all()

  def test_tfconfig_shrink(self):
    num_workers = 3
    w_list = [f"localhost:555{ii}" for ii in range(num_workers)]
    for ii in range(num_workers):
      params = ElasticParams(f"localhost:555{ii}", "2:3")
      self._start_node(self._run_elastic_tfconfig_test, {"params":params, "sleep":15})
    time.sleep(5)
    self._end_node(1)
    self.join_all()

  def _setup_mnist(self, strategy):
    import tensorflow as tf
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the range [0, 255].
    # We need to convert them to float32 with values in the range [0, 1]
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(60000).repeat().batch(256)
    
    with strategy.scope():
      model = tf.keras.Sequential([
          tf.keras.Input(shape=(28, 28)),
          tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
          tf.keras.layers.Conv2D(32, 3, activation='relu'),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dense(10)
      ])
      model.compile(
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
          metrics=['accuracy'])
    
    return model, train_dataset

  def _run_mnist_shrink(self, params, sleep):
    handler = rdzv.TFEOrchestratorHandler(SERVER_ADDRESS, params.min_nodes, params.max_nodes)
    prev_tfconfig = handler.GetClusterSpec(params.address, False)
    
    os.environ["TF_CONFIG"] = json.dumps(prev_tfconfig)
    prev_tfconfig = os.environ["TF_CONFIG"]
    print(prev_tfconfig, flush=True)
    
    strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()
    timings = []
    model, td = self._setup_mnist(strategy)
    for ii in range(3):
      start = time.time()
      model.fit(td, epochs=1, steps_per_epoch=10)
      timings.append(time.time() - start)

    
    time.sleep(sleep)
    
    tfconfig = handler.GetClusterSpec(params.address, False)
    os.environ["TF_CONFIG"] = json.dumps(tfconfig)
    assert prev_tfconfig != os.environ["TF_CONFIG"], (prev_tfconfig, os.environ["TF_CONFIG"])
    print(os.environ["TF_CONFIG"], flush=True)
    
    strategy.extended.update_cluster()
    time.sleep(3)
    timings2 = []
    for ii in range(3):
      start = time.time()
      model.fit(td, epochs=1, steps_per_epoch=10)
      timings2.append(time.time() - start)
    print(f"Before update {timings}, After update {timings2}")



  def test_mnist_shrink(self):
    num_workers = 3
    w_list = [f"localhost:555{ii}" for ii in range(num_workers)]
    for ii in range(num_workers):
      params = ElasticParams(f"localhost:555{ii}", "2:3")
      self._start_node(self._run_mnist_shrink, {"params":params, "sleep":15})
    time.sleep(20)
    self._end_node(1)
    self.join_all()

    

  

if __name__ == '__main__':
  test.main()