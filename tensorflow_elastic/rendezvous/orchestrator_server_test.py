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
import grpc
import sys

#os.environ["TF_CPP_VMODULE"] ="collective_param_resolver_distributed=2,collective_param_resolver_local=2,collective_ops=3,base_collective_executor=3,hierarchical_tree_broadcaster=3"
#os.environ["TF_CPP_MIN_VLOG_LEVEL"] ="0"
from multiprocessing import Pool, Process
import subprocess
import time
from tensorflow.python.platform import test
from tensorflow_elastic.rendezvous.orchestrator_server import serve
import tensorflow_elastic.orchestrator_pb2_grpc as orchestrator_pb2_grpc
import tensorflow_elastic.orchestrator_pb2 as orchestrator_pb2
import tensorflow_elastic.rendezvous.orchestrator_api as orch_api

SERVER_ADDRESS = 'localhost:50051'

def GetClusterSpec(handler, address, reset, sleep=0):
  time.sleep(sleep)
  return handler.GetClusterSpec(address, reset)

def GetWaitingNodes(handler, address, sleep=0):
  time.sleep(sleep)
  return handler.GetWaitingNodes(address)

def Barrier(handler, address, tag, timeout, sleep=0):
  time.sleep(sleep)
  return handler.Barrier(address, tag, timeout)

def Synchronize(handler, address, tag, data, timeout, sleep=0):
  time.sleep(sleep)
  return handler.Synchronize(address, tag, data, timeout)


class TFEOrchestratorTest(test.TestCase):
  def setUp(self):
    # Setup the etcd server
    self._t = Process(target=serve)
    self._t.start()
    time.sleep(2)

  def tearDown(self):
    self._t.terminate()
  
  def test_bringup(self):
    time.sleep(10)
    assert self._t.is_alive()

  def test_single_connect(self):
    handler = orch_api.TFEOrchestratorHandler(SERVER_ADDRESS, 1, 1)
    c_spec = handler.GetClusterSpec("localhost:50052", False)
    assert len(c_spec["cluster"]["worker"]) == 1, str(c_spec)
    
  def test_multi_connect(self):
    args = []
    num_workers = 3
    handler = orch_api.TFEOrchestratorHandler(SERVER_ADDRESS, 3, 3)
    for ii in range(num_workers):
      args.append((handler, f"localhost:{50051+ii}", False))

    pool = Pool(num_workers)
    results = pool.starmap(GetClusterSpec, args)

    assert len(results[0]["cluster"]["worker"]) == num_workers, results[0]

  def test_waiting_workers(self):
    args = []
    num_workers = 3
    min_nodes = 2
    max_nodes = 3
    handler = orch_api.TFEOrchestratorHandler(SERVER_ADDRESS, min_nodes, max_nodes)
    for ii in range(min_nodes):
      args.append((handler, f"localhost:{50051+ii}", False))

    pool = Pool(num_workers)
    results = pool.starmap(GetClusterSpec, args)
    
    assert len(results[0]["cluster"]["worker"]) == min_nodes, results[0]
    time.sleep(5)
    p = Process(target=GetClusterSpec, args=(handler, "localhost:50054", False))
    p.start()
    wait_nodes = GetWaitingNodes(handler, "localhost:50051")
    assert wait_nodes == (num_workers-max_nodes), f"Waiting nodes {wait_nodes} num_workers {num_workers} max_nodes {max_nodes}"
    p.terminate()
    p.join()


  # def test_excess_workers(self):
  #   args = []
  #   num_workers = 3
  #   min_nodes = 2
  #   max_nodes = 3
  #   for ii in range(max_nodes):
  #     args.append((f"localhost:{50051+ii}", min_nodes, max_nodes, False))

  #   pool = Pool(num_workers)
  #   results = pool.starmap(GetClusterSpec, args)
    
  #   assert len(results[0]["cluster"]["worker"]) == max_nodes, results[0]
  #   time.sleep(10)
  #   p = Process(target=GetClusterSpec, args=("localhost:50054", min_nodes, max_nodes, False))
  #   p.start()
  #   wait_nodes = GetWaitingNodes()
  #   assert wait_nodes == 0, f"Waiting nodes {wait_nodes} num_workers {num_workers} max_nodes {max_nodes}"
  #   p.terminate()
  #   p.join()

  def test_reset_workers(self):
    args = []
    num_workers = 3
    min_nodes = 2
    max_nodes = 3
    handler = orch_api.TFEOrchestratorHandler(SERVER_ADDRESS, min_nodes, max_nodes)
    for ii in range(min_nodes):
      args.append((handler, f"localhost:{50051+ii}", False))

    pool = Pool(min_nodes)
    results = pool.starmap(GetClusterSpec, args)
    
    assert len(results[0]["cluster"]["worker"]) == min_nodes, results[0]
    
    p = Process(target=GetClusterSpec, args=(handler, "localhost:50054", False))
    p.start()
    time.sleep(2)
    wait_nodes = GetWaitingNodes(handler,"localhost:50051")
    assert wait_nodes == 1, f"Waiting nodes {wait_nodes} num_workers {num_workers} max_nodes {max_nodes}"

    pool.close()
    print("Finished Waiting Nodes",flush=True)
    args = []
    for ii in range(min_nodes):
      args.append((handler, f"localhost:{50051+ii}", True))

    pool = Pool(num_workers)
    results = pool.starmap(GetClusterSpec, args)
    print(results, flush=True)
    p.join()

    assert len(results[0]["cluster"]["worker"]) == max_nodes, results[0]
    
    
  def test_synchronize_workers(self):

    num_workers = 3
    tag="test"
    args = []
    setup_args = []
    final = {}
    handler = orch_api.TFEOrchestratorHandler(SERVER_ADDRESS, num_workers, num_workers)
    for ii in range(num_workers):
      address = f"localhost:{500051+ii}"
      data = json.dumps({ii:address})
      final[address]=json.loads(data)
      args.append((handler, address, tag, data, 10))
      setup_args.append((handler, address, False))

    pool = Pool(num_workers)
    results = pool.starmap(GetClusterSpec, setup_args)
    results = pool.starmap(Synchronize, args)

    print(results)
    success = [ret[0] for ret in results]
    error_msgs = [(ret[2]=="") for ret in results]
    data = {key: json.loads(results[0][1][key]) for key in results[0][1]}
    assert all(success), success
    assert all(error_msgs), error_msgs
    assert len(results), results
    assert data == final, (data, final)

  def test_fail_synchronize_workers(self):

    num_workers = 3
    tag="test"
    args = []
    setup_args = []
    handler = orch_api.TFEOrchestratorHandler(SERVER_ADDRESS, num_workers, num_workers)
    for ii in range(num_workers):
      address = f"localhost:{500051+ii}"
      data = json.dumps({ii:address})
      args.append((handler, address, tag, data, 10, ii*6))
      setup_args.append((handler, address, False))

    pool = Pool(num_workers)
    results = pool.starmap(GetClusterSpec, setup_args)
    try:
      results = pool.starmap(Synchronize, args)
    except ValueError as e:
      assert "Timeout reached" in str(e), str(e)
    

    # print(results)
    # success = [ret[0] for ret in results]
    # error_msgs = [(ret[2].startswith("Timeout reached")) for ret in results]
    # data = {key: json.loads(results[0][1][key]) for key in results[0][1]}
    # assert not any(success), success
    # assert all(error_msgs), error_msgs
    # assert len(results), results
    # assert data == {}, data


  def test_barrier_workers(self):

    num_workers = 3
    tag="test"
    args = []
    setup_args = []
    handler = orch_api.TFEOrchestratorHandler(SERVER_ADDRESS, num_workers, num_workers)
    for ii in range(num_workers):
      address = f"localhost:{500051+ii}"
      args.append((handler, address, tag, 10, ii*3))
      setup_args.append((handler, address, False))

    pool = Pool(num_workers)
    results = pool.starmap(GetClusterSpec, setup_args)
    results = pool.starmap(Barrier, args)

    
    print(results)
    success = [ret[0] for ret in results]
    error_msgs = [(ret[1]=="") for ret in results]
    
    assert all(success), success
    assert all(error_msgs), error_msgs
    assert len(results), results
    
  def test_fail_barrier_workers(self):

    num_workers = 3
    tag="test"
    args = []
    setup_args = []
    handler = orch_api.TFEOrchestratorHandler(SERVER_ADDRESS, num_workers, num_workers)
    for ii in range(num_workers):
      address = f"localhost:{500051+ii}"
      args.append((handler, address, tag, 10, ii*6))
      setup_args.append((handler, address, False))

    pool = Pool(num_workers)
    results = pool.starmap(GetClusterSpec, setup_args)
    try:
      results = pool.starmap(Barrier, args)
    except ValueError as e:
      assert "Timeout reached" in str(e), str(e)

    
  # def test_fail_barrier_workers_indefinite(self):

  #   num_workers = 3
  #   tag="test"
  #   args = []
  #   setup_args = []
  #   handler = orch_api.TFEOrchestratorHandler(SERVER_ADDRESS, num_workers, num_workers)
  #   for ii in range(num_workers):
  #     address = f"localhost:{500051+ii}"
  #     args.append((handler, address, tag, -1, ii*6))
  #     setup_args.append((handler, address, False))

  #   pool = Pool(num_workers)
  #   results = pool.starmap(GetClusterSpec, setup_args)
  #   try:
  #     results = pool.starmap(Barrier, args)
  #   except ValueError as e:
  #     assert "Timeout reached" in str(e), str(e)

  
    

    
    # print(results)
    # success = [ret[0] for ret in results]
    # error_msgs = [(ret[1].startswith("Timeout reached")) for ret in results]
    
    # assert not any(success), success
    # assert all(error_msgs), error_msgs
    # assert len(results), results
  

  def test_shutdown(self):
    args = []
    args2 = []
    args3 = []
    
    num_workers = 3
    handler = orch_api.TFEOrchestratorHandler(SERVER_ADDRESS, 3, 3)
    for ii in range(num_workers):
      address =f"localhost:{50051+ii}"
      data = json.dumps({ii:address})
      args.append((handler, address, False))
      args2.append((handler, address, "tag", 10, 1))
      args3.append((handler, address, "tag", data, 10, 1))
    
    pool = Pool(num_workers)
    results = pool.starmap(GetClusterSpec, args)

    assert len(results[0]["cluster"]["worker"]) == num_workers, results[0]
    time.sleep(5)
    end_time = handler.ShutDown()
    assert end_time != "", end_time
    print(end_time, flush=True)

    results = pool.starmap(GetClusterSpec, args)
    assert results[0] == {}, results[0]

    try:
      _ = pool.starmap(Barrier, args2)
    except ValueError as e:
      assert "shutdown" in str(e).lower(), str(e).lower()

    try:
      _ = pool.starmap(Synchronize, args3)
    except ValueError as e:
      assert "shutdown" in str(e).lower(), str(e).lower()

    try:
      _ = GetWaitingNodes(handler,"localhost:50051")
    except ValueError as e:
      assert "shutdown" in str(e).lower(), str(e).lower()



    

if __name__ == '__main__':
  test.main()