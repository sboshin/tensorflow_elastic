"""Tests for No worker restart"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import os
#os.environ["TF_CPP_VMODULE"] ="collective_param_resolver_distributed=2,collective_param_resolver_local=2,collective_ops=3,base_collective_executor=3,hierarchical_tree_broadcaster=3"
#os.environ["TF_CPP_MIN_VLOG_LEVEL"] ="1"

import time, json
from tensorflow.python.platform import test
from tensorflow_elastic.distributed.launch import main
from tensorflow_elastic.rendezvous.orchestrator_server import serve

import tensorflow_elastic.rendezvous.orchestrator_api as rdzv
import tensorflow_elastic.tensorflow
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow_elastic.tests import utils
from tensorflow_elastic.tests.bin.resnet_mwms_ctl_new_design import main as ctl_main
import tensorflow_elastic.rendezvous.orchestrator_api as orch_api


class ElasticCTL(utils.MultiWorkerTestBase, parameterized.TestCase):

  def test_ctl_shrink(self):
    num_workers = 3
    self.start_orchestrator()
    self.num_workers = num_workers
    for ii in range(num_workers):
      self._start_node(ctl_main)
    time.sleep(90)
    self._end_node(2)
    self.join_all()

  def test_ctl_grow(self):
    num_workers = 2
    self.start_orchestrator()
    self.num_workers = num_workers
    for ii in range(num_workers):
      self._start_node(ctl_main)
    time.sleep(90)
    self._start_node(ctl_main)
    self.join_all()

  def test_ctl_shrink_grow(self):
    num_workers = 3
    self.start_orchestrator()
    self.num_workers = num_workers
    for ii in range(num_workers):
      self._start_node(ctl_main)
    time.sleep(90)
    self._end_node(2)
    time.sleep(90)
    self._start_node(ctl_main)
    self.join_all()

  def _set_tfconfig(self, tf_config):
    os.environ["TF_CONFIG"] = json.dumps(tf_config)

  def _simple_shrink_test(self, **kwargs):
    
    handler = self._get_handler(1, 3)
    local_address = f"localhost:{50050+kwargs['proc_id']}"
    cluster_spec = handler.GetClusterSpec(local_address, False)
    print(cluster_spec, flush=True)
    self._set_tfconfig(cluster_spec)
    strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()
    strategy.extended._stop_check_health_thread()
    self.Barrier(kwargs["proc_id"])
    
    print("OLD: ", cluster_spec)
    cluster_spec = handler.GetClusterSpec(local_address, False)
    self._set_tfconfig(cluster_spec)
    print("NEW: ", os.environ["TF_CONFIG"], flush=True)
    strategy.extended.update_cluster()
    print(f"{kwargs['proc_id']} Test finished Successfully")
    return 0

  def _simple_grow_test(self, **kwargs):
    handler = self._get_handler(1, 3)
    local_address = f"localhost:{50050+kwargs['proc_id']}"
    cluster_spec = handler.GetClusterSpec(local_address, False)
    print(cluster_spec, flush=True)
    time.sleep(2)
    self._set_tfconfig(cluster_spec)
    strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()
    if(kwargs["proc_id"] > 1):
      return
    strategy.extended._stop_check_health_thread()
    self.Barrier(kwargs["proc_id"])
    kwargs["wait"].wait()
    
    print("OLD: ", cluster_spec, flush=True)
    cluster_spec = handler.GetClusterSpec(local_address, False)
    self._set_tfconfig(cluster_spec)
    print("NEW: ", os.environ["TF_CONFIG"], flush=True)
    strategy.extended.update_cluster()
    print(f"{kwargs['proc_id']} Test finished Successfully")
    
    
  def _get_handler(self, minN, maxN):
    tf_config = json.loads(os.environ["TF_CONFIG"])
    server_address = tf_config["cluster"]["orchestrator"][0]
    return orch_api.TFEOrchestratorHandler(server_address, minN, maxN)

  def test_cluster_shrink(self):

    num_workers = 3
    self.num_workers = num_workers
    self.start_orchestrator()
    self._create_barrier_array(num_workers)
    for ii in range(num_workers):
      self._start_node(self._simple_shrink_test)
    self.ClientBarrier(num_workers)
    self._end_node(2)
    self.wait.set()
    self.join_all()
    print("Test finished successfully")
    return 0

  def test_cluster_grow(self):
    num_workers = 2
    self.num_workers = num_workers
    self.start_orchestrator()
    self._create_barrier_array(num_workers)
    for ii in range(num_workers):
      self._start_node(self._simple_grow_test)
    
    self.ClientBarrier(num_workers)
    
    time.sleep(1)
    self.num_workers +=1
    self._start_node(self._simple_grow_test)
    
    self.join_all()

  def test_resnetctlElastic(self):
    self._reset_backupdir()
    
    num_workers = 2
    self.start_orchestrator()
    self.num_workers = num_workers
    for ii in range(num_workers):
      self._start_node(ctl_main)
    time.sleep(40)
    self._start_node(ctl_main)
    time.sleep(40)
    self._end_node(2)

    self.join_all()

  def _reset_backupdir(self):
    tmp_dir = "/tmp/backup"
    if(os.path.isdir(tmp_dir)):
     import shutil
     shutil.rmtree(tmp_dir)
    

if __name__ == "__main__":
  test.main()
