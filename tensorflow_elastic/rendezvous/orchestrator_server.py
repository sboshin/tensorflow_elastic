from concurrent import futures
import time
import math
import logging
import json
import queue

import grpc

import tensorflow_elastic.orchestrator_pb2 as orchestrator_pb2
import tensorflow_elastic.orchestrator_pb2_grpc as orchestrator_pb2_grpc
from tensorflow_elastic.utils.logging import get_logger
import threading
NODE_GATHER_TIMEOUT = 30
HEALTH_CHECK_TIMEOUT = 10

log = get_logger("default")

def printf(*args, **kwargs):
  print(*args, **kwargs, flush=True)

class TFEOrchestratorServicer(orchestrator_pb2_grpc.TFEOrchestratorServicer):

  def __init__(self):
    #To ensure the same setup, we will enforce all registration to have the same parameters
    # Min Nodes
    # Max Nodes

    self._current_id = 0
    self._next_id = 0
    self._workers = {0:{}}
    self._params = None
    self._lock = threading.Lock()
    self._param_lock = threading.Lock()
    self._waiting_nodes = 0
    self._start_time = None
    self._end_time = None
    self._curr_nodes = 0
    self._ret_nodes = 0
    self._node_lock = threading.Lock()
    self._node_event = threading.Event()
    self._node_event.clear()
    self._node_prep = threading.Event()
    self._node_prep.set()
    self._min_timer_event = threading.Event()
    self._node_cond = threading.Condition()
    self._state = ""
    self._bar_lock = threading.Lock()
    self._sync_lock = threading.Lock()
    self._start_bar = False
    self._wait_thread = None
    self._working_id = 0
    self._start_sync = None
    self._failed_lease = False
    self._address_index = {}
    self._max_index = 0
    self._node_queue = queue.Queue()
    #Testing
    #self._address_index["localhost:50050"] = 2
    #self._address_index["localhost:50051"] = 0
    #self._address_index["localhost:50052"] = 1
    #self._max_index = 3
  
  def _create_sparse_worker_dict(self, workers, address):
    ret = {}
    my_index = None
    print(workers, address, flush=True)
    for worker in workers:
      if(worker not in self._address_index):
        self._address_index[worker] = self._max_index
        self._max_index +=1

      ret[self._address_index[worker]] = worker
      
      if(worker == address):
        my_index = self._address_index[address]
    print(ret, my_index, flush=True)
    return ret, my_index



  def UniqueIndex(self, request, context):
    worker = request.address
    self._node_lock.acquire()
    if(worker not in self._address_index):
        self._address_index[worker] = self._max_index
        self._max_index +=1
    self._node_lock.release()
    return orchestrator_pb2.UniqueIndexResponse(unique_index=str(self._address_index[worker]))


  def _create_cluster_spec(self, cid, request):
    cluster_spec_template = {"cluster":{"worker":{}}, "task":{"index": None, "type":"worker"}}
    #cluser_spec_template["cluster"]["worker"] = list(self._workers[self._current_id].keys())
    worker_dict, my_index = self._create_sparse_worker_dict(list(self._workers[cid].keys()), request.address)
    cluster_spec_template["cluster"]["worker"] = worker_dict
    cluster_spec_template["task"]["index"] = my_index
    ret = json.dumps(cluster_spec_template)
    log.info(ret)
    log.info(self._workers)
    return ret

  def _check_shutdown(self):
    if(self._state == "shutdown"):
      return True
    else:
      return False


  def _verify_params(self, request):
    if(self._params is None):
      self._param_lock.acquire()
      if(self._params is None):
        self._params = {"min_nodes":request.min_nodes, "max_nodes":request.max_nodes}
        self._min_barrier = threading.Barrier(request.min_nodes)
        self._start_time = time.time()
      self._param_lock.release()
    
    #Verify params are the same
    v1 = self._params["min_nodes"] == request.min_nodes
    v2 = self._params["max_nodes"] == request.max_nodes
    return v1 and v2

  def GetClusterSpec(self, request, context):
    if(self._check_shutdown()):
      #return empty cluster spec
      return orchestrator_pb2.ClusterSpec(cluster_spec="{}")

    if(not self._verify_params(request)):
      log.warning(f"Request {request} doesn't match params {self._params}")
      return orchestrator_pb2.ClusterSpec(cluster_spec="{}")

    log.warning(f"Request is {request}")

    if(request.reset):
      #Delete address map
      log.warning(f"Resetting address index map")
      self._address_index = {}
      self._max_index = 3
      self._address_index["localhost:50053"] = 2
      self._address_index["localhost:50054"] = 0
      self._address_index["localhost:50055"] = 1

    
    log.warning(f"{request.address} waiting for prep")
    self._node_prep.wait()
    log.warning(f"{request.address} update and check")
    curr_id = self._update_and_check_nodes(request.address)
    log.warning(f"{request.address} Node event {self._node_event.is_set()}")
    log.warning(f"{request.address} waiting for event")
    self._node_event.wait()
    
    self._node_queue.put(request.address)
    
    self._min_timer_event.clear()
    
    log.warning(f"{request.address} creating cspec")
    with self._node_lock:
      c_spec = orchestrator_pb2.ClusterSpec(cluster_spec=self._create_cluster_spec(curr_id, request))
    
    log.warning(f"{request.address} created cspec")
    self._wait_thread = None #should we delete this??
    self._working_id = curr_id
    
    #Don't clear until all nodes are accounted for.
    with self._node_lock:
      cur_num = len(self._workers[self._working_id])
      log.warning(f"{request.address} Current number of nodes is {cur_num}")

    while(self._node_queue.qsize() < cur_num):
      with self._node_lock:
        cur_num = len(self._workers[self._working_id])
        log.warning(f"{request.address} Current number of nodes is {cur_num}")
      time.sleep(1)
    self._node_event.clear()
    self._node_prep.set()

    log.warning(f"{request.address} finished GetClusterSpec")
    return c_spec

  def _node_event_setup(self):
    self._node_prep.clear()
    self._node_event.set()
    self._current_id +=1
    self._workers[self._current_id] = {}
    

  def _min_timer(self):
    print("Going to sleep", flush=True)
    time.sleep(NODE_GATHER_TIMEOUT)
    print("waking up", flush=True)
    if (self._min_timer_event.is_set()):
      self._node_event_setup()

  def _start_wait_thread(self):
    printf(self._wait_thread)
    self._min_timer_event.set()
    if(self._wait_thread is None):
      self._wait_thread = threading.Thread(target=self._min_timer)
      self._wait_thread.start()



  def _update_and_check_nodes(self, address):
    #Check number of nodes, if nodes >= min nodes and < max_nodes
    # set node pass event
    with self._node_lock:
      cur_nodes = len(self._workers[self._current_id])
      cur_id = self._current_id
      
      if(cur_nodes <= self._params["max_nodes"]):
        self._workers[self._current_id].update({address:0})
        cur_nodes = len(self._workers[self._current_id])

      print(f"Current nodes cur_nodes at {self._current_id} is {cur_nodes}", flush=True)
      if(cur_nodes == self._params["min_nodes"]):
        #Start the timer
        printf("Attempting to start wait thread")
        self._start_wait_thread()
        pass

      
      if(cur_nodes == self._params["max_nodes"]):
        self._node_event_setup()

    return cur_id


  def GetWaitingNodes(self, request, context):
    if(self._check_shutdown()):
      return orchestrator_pb2.WaitingNodes(num_waiting_nodes=0, error_msg="Server has shutdown") 

    # We will use this as a health check.
    # Healthy nodes should be checking this regularly. If a node fails to check this in HEALTH_CHECK_TIMEOUT
    # We assume this node is dead
    
    #First verify we are in the list of workers
    #log.info(f"Getting waiting nodes, {request.address} : {self._workers[self._working_id]}")
    address = request.address
    if(address not in self._workers[self._working_id]):
      if(address in self._workers[self._current_id]):
        #future id, and we can just return
        log.info(f"Returning passed. {address}: {self._workers[self._current_id]}")
        return orchestrator_pb2.WaitingNodes(num_waiting_nodes=0, error_msg="") 

      #Return an error message if we fail
      log.info(f"Returning failed. {address}: {self._workers[self._working_id]}")
      return orchestrator_pb2.WaitingNodes(num_waiting_nodes=self._waiting_nodes, error_msg=f"{address} not in current worker {self._workers[self._working_id]}") 
    
    self._workers[self._working_id][address] = time.time()

    if(self._waiting_nodes > 0):
      return orchestrator_pb2.WaitingNodes(num_waiting_nodes=self._waiting_nodes, error_msg="")
    
    #If we are already waiting for nodes (being added) we can return
    #now check for Health check failures

    fail = [self._workers[self._working_id][x] + HEALTH_CHECK_TIMEOUT > time.time() for x in self._workers[self._working_id]]
    if(fail.count(True) > 0):
      self._failed_lease = True
    
    num_waiting = len(self._workers[self._current_id])
    return orchestrator_pb2.WaitingNodes(num_waiting_nodes=num_waiting, error_msg="")  


  def Synchronize(self, request, context):
    # Blocking call/timeout that will synchronize data, and dump out as json string
    def ret_val(ret_bool: bool, data: str, msg=""):
      self._start_sync = False
      return orchestrator_pb2.SyncResponse (success=ret_bool, data=data, error_msg=msg)

    if(self._check_shutdown()):
      return ret_val(False, "{}", "Server has shutdown")
    
    end_time = time.time() + (2^62-1) if(request.timeout <= 0) else time.time() + request.timeout

    
    self._sync_lock.acquire()
    if(not self._start_sync):
      self._data_sync = {}
      self._start_sync = True
    
    if(request.address not in self._workers[self._working_id]):
      self._sync_lock.release()
      return ret_val(False, "{}", f"{request.address} not in current list of workers {str(self._workers)}")

    self._data_sync[request.address] = request.data
    self._sync_lock.release()

    while (time.time() <= end_time):
      sync_len = len(self._data_sync)
      if(sync_len == len(self._workers[self._working_id])):
        return ret_val(True, json.dumps(self._data_sync))
      elif(sync_len > len(self._workers[self._working_id])):
        self._start_sync = False
        raise ValueError(f"{sync_len} num workers synchronizing > {len(self._workers[self._working_id])} ")
      else:
        time.sleep(1)
        log.warning(f"{sync_len}")
    if(len(self._data_sync) == len(self._workers[self._working_id])):
        return ret_val(True, json.dumps(self._data_sync))
    else:
      self._data_sync.pop(request.address, None)
      return ret_val(False, "{}", f"Timeout reached {request.timeout}")

  def Barrier(self, request, context):
    def ret_val(ret_bool: bool, msg=""):
      self._start_bar = False
      return orchestrator_pb2.BarrierResponse (success=ret_bool, error_msg=msg)

    if(self._check_shutdown()):
      return ret_val(False, "Server has shutdown")

    end_time = time.time() + (2**62-1) if(request.timeout <= 0) else time.time() + request.timeout

    log.warning(f"Timeout ends {end_time - time.time()}s")

    self._bar_lock.acquire()
    if(not self._start_bar):
      self._data_bar = {}
      self._start_bar = True
    
    if(request.address not in self._workers[self._working_id]):
      self._bar_lock.release()
      return ret_val(False, f"{request.address} not in current list of workers {str(self._workers[self._working_id])}")

    self._data_bar[request.address]=1
    self._bar_lock.release()
    
    while (time.time() <= end_time):
      bar_len = len(self._data_bar)
      log.warning(f"{self._workers[self._working_id]} bar_len {bar_len} working_id {self._working_id}")
      if(bar_len == len(self._workers[self._working_id])):
        return ret_val(True, "")
      elif(bar_len > len(self._workers[self._working_id])):
        self._start_bar = False
        raise ValueError(f"{bar_len} num workers synchronizing > {len(self._workers[self._working_id])} ")
      else:
        time.sleep(1)
    if(len(self._data_bar) == len(self._workers[self._working_id])):
        return ret_val(True, "")
    else:
      self._data_bar.pop(request.address, None)
      return ret_val(False, f"Timeout reached {request.timeout}")

  def ShutDown(self, request, context):
    self._param_lock.acquire()
    if(self._end_time is None):
      self._end_time = time.time()
      self._state = "shutdown"
    self._param_lock.release()
    end_timing_string = f"Server started {self._start_time} Server ended {self._end_time} \nTotal time taken is {self._end_time - self._start_time}"
    log.warning(end_timing_string)
    def ret_val(ret_str: str):
      return orchestrator_pb2.ShutDownResponse(end_time=ret_str)

    return ret_val(end_timing_string)

def serve(port="50051"):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))
    orchestrator_pb2_grpc.add_TFEOrchestratorServicer_to_server(
        TFEOrchestratorServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()
    


if __name__ == '__main__':
    #logging.basicConfig(
    #    level=logging.INFO, format="[%(levelname)s] %(asctime)s %(module)s: %(message)s", force=True
    #)
    serve()

