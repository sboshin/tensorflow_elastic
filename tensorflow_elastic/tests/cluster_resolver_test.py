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
  def __init__(self, nnodes="1:3", nproc_per_node=1, etcd_endpoint=5379, run_id=1, script="bin/tf_config_test.py", args=["--expected_tf_config=None"]):
    self.nnodes=nnodes
    self.nproc_per_node=nproc_per_node
    self.etcd_endpoint = etcd_endpoint
    self.run_id = run_id
    self.script = script
    self.args = args


class ElasticTensorflowTFConfigTest(test.TestCase):
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
    

  def test_tfconfig(self):
    
    start_count = 2
    for ii in range(start_count):
      params = ElasticParams(args=[], nnodes=start_count)
      self._start_node(self._run_in_launch, {"params":params})
    self.join_all()

  def test_tfconfig_growth(self):
    
    start_count = 2
    nnodes = "2:3"
    args = ["--sleep=70"]
    for ii in range(start_count):
      params = ElasticParams(args=args, nnodes=nnodes)
      self._start_node(self._run_in_launch, {"params":params})
    time.sleep(40)
    start_count +=1
    args = ["--sleep=70"]
    params = ElasticParams(args=args, nnodes=nnodes)
    self._start_node(self._run_in_launch, {"params":params})

    time.sleep(10)

    self.join_all()

  def test_tfconfig_shrink(self):
    
    start_count = 3
    nnodes = "2:3"
    args = ["--sleep=100"]
    for ii in range(start_count):
      params = ElasticParams(args=args, nnodes=nnodes)
      self._start_node(self._run_in_launch, {"params":params})
    time.sleep(25)
    
    
    #for ii in range(start_count):
    #  self._end_node(0)
    self._end_node(0)
    #time.sleep(40)

    self.join_all()

  def test_Collectiveallreduce_variable(self):
    start_count = 2
    nnodes = "2:3"
    args = ["--sleep=60"]
    p_args = {"args":args, "nnodes":nnodes, "script":"bin/num_workers_test.py"}
    for ii in range(start_count):
      params = ElasticParams(**p_args)
      self._start_node(self._run_in_launch, {"params":params})
    time.sleep(55)


    params = ElasticParams(**p_args)
    self._start_node(self._run_in_launch, {"params":params})

    time.sleep(25)

    self._end_node(0)

    time.sleep(55)

    self.join_all()

  def _reset_backupdir(self):
    tmp_dir = "/tmp/backup"
    if(os.path.isdir(tmp_dir)):
     import shutil
     shutil.rmtree(tmp_dir)
    
  def test_mnistElastic(self):
    self._reset_backupdir()
    start_count = 2
    nnodes = "2:3"
    args = ["--epochs=30"]
    p_args = {"args":args, "nnodes":nnodes, "script":"bin/mnist_mwms.py"}
    for ii in range(start_count):
      params = ElasticParams(**p_args)
      self._start_node(self._run_in_launch, {"params":params})
    time.sleep(55)


    params = ElasticParams(**p_args)
    self._start_node(self._run_in_launch, {"params":params})

    time.sleep(25)

    self._end_node(2)

    self.join_all()

  def test_resnetElastic(self):
    self._reset_backupdir()
    start_count = 2
    nnodes = "2:3"
    args = ["--epochs=10"]
    p_args = {"args":args, "nnodes":nnodes, "script":"bin/resnet_mwms.py"}
    for ii in range(start_count):
      params = ElasticParams(**p_args)
      self._start_node(self._run_in_launch, {"params":params})
    time.sleep(75*3)


    params = ElasticParams(**p_args)
    self._start_node(self._run_in_launch, {"params":params})

    time.sleep(75*3)

    self._end_node(2)

    self.join_all()

if __name__ == '__main__':
  test.main()