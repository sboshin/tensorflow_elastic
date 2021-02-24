import os
import sys
import multiprocessing as mp
import psutil
import signal
import shutil
import time
import json

from multiprocessing import Process, Queue, set_start_method

from tensorflow.python.platform import test
from tensorflow_elastic.distributed.launch import main as launch

def path(script):
  return os.path.join(os.path.dirname(__file__), script)


def kill_proc_tree(pid,
                   sig=signal.SIGTERM,
                   include_parent=True,
                   timeout=None,
                   on_terminate=None):
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
  gone, alive = psutil.wait_procs(children,
                                  timeout=timeout,
                                  callback=on_terminate)
  print(f"Gone {gone} Alive {alive}", flush=True)
  return (gone, alive)


class MultiWorkerTestBase(test.TestCase):

  def Barrier(self, proc_id):
    self.barrier_array[proc_id] = 1
    #while (not self.wait.is_set()):
    #   print("Spin wait in barrier")
    #   time.sleep(.1)
    #print(f"Proc id {proc_id} setting to 1 and waiting", flush=True)
    #self.wait.wait()
    #print(f"Finished waiting for {proc_id}")

  def ClientBarrier(self, parties):

    while (1):
      cur_mem = 0
      for ii in self.barrier_array:
        if (ii == 1):
          cur_mem += 1
      if (cur_mem == parties):
        self.wait.set()
        print(self.barrier_array, flush=True)
        self.wait.wait()
        return
      

  def _create_barrier_array(self, num_workers):
    self.barrier_array = mp.Array('B', [0] * num_workers)

  def setUp(self):
    super(MultiWorkerTestBase, self).setUp()
    self.barrier_array = None  #mp.Array('B', [])
    self.wait = mp.Event()
    self.wait.clear()
    self._procs = []
    self._exclude_procs = []
  
  def tearDown(self):
    if(hasattr(self, "orchestrator")):
      proc = self.orchestrator
      kill_proc_tree(proc.pid, timeout=2, include_parent=True)
      #time.sleep(1)
      proc.terminate()
      proc.join()
    super(MultiWorkerTestBase, self).tearDown()


  def join_all(self):
    for ii, proc in enumerate(self._procs):
      self._procs[ii].join()
      print(self._procs[ii])
      if (self._procs[ii].exitcode != 0 and ii not in self._exclude_procs):
        raise RuntimeError(
            "Process %d failed with exitcode %d but wasn't in exclude list" %
            (ii, self._procs[ii].exitcode))
  def _wrap(self, fn, *args, **kwargs):
    cluster_spec = {
        "cluster": {
            "worker": []
        },
        "task": {
            "index": kwargs["proc_id"],
            "type": "worker"
        }
    }
    if(self.orchestrator): #orchestrator has started
      cluster_spec["cluster"]["orchestrator"] = [self.SERVER_ADDRESS]
    
    assert self.num_workers
    cluster = [f"localhost:{5000+ii}" for ii in range(self.num_workers)]
    cluster_spec["cluster"]["worker"] = cluster
    os.environ["TF_CONFIG"] = json.dumps(cluster_spec)
    print(f"Setting TF_CONFIG to {os.environ['TF_CONFIG']}")
    return fn(*args, **kwargs)

  def start_orchestrator(self):
    self.SERVER_ADDRESS = 'localhost:55555'
    args = [
            f"--standalone",
            f"--rdzv_endpoint={self.SERVER_ADDRESS}",
           "ls"]
    self.orchestrator = Process(target=launch, args=(args,))
    self.orchestrator.start()
    
  def _start_node(self, target, params={}):
    #Use launc to start the process
    params.update({
        "barrier_array": self.barrier_array,
        "wait": self.wait,
        "proc_id": len(self._procs)
    })
    if(hasattr(self, "num_workers")):
      t = Process(target=self._wrap, args=(target,), kwargs=params)
    else:
      t = Process(target=target, kwargs=params)
    t.start()
    print("Starting ", str(target), f" at {t.pid}")
    self._procs.append(t)

  def _end_node(self, node_num):
    #Use launc to start the process
    print(f"Terminating process {node_num}", flush=True)
    proc = self._procs.pop(node_num)
    print(f"Terminating proc tree {proc.pid}")
    kill_proc_tree(proc.pid, timeout=2, include_parent=True)
    #time.sleep(1)
    proc.terminate()
    proc.join()
    print(f"After terminate, {proc.is_alive()}", flush=True)


class ElasticParams(object):

  def __init__(self,
               nnodes="1:3",
               nproc_per_node=1,
               etcd_endpoint=5379,
               run_id=1,
               script="bin/tf_config_test.py",
               args=["--expected_tf_config=None"]):
    self.nnodes = nnodes
    self.nproc_per_node = nproc_per_node
    self.etcd_endpoint = etcd_endpoint
    self.run_id = run_id
    self.script = script
    self.args = args
