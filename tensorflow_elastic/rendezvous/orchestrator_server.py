from concurrent import futures
import time
import math
import logging
import json

import grpc

import tensorflow_elastic.orchestrator_pb2 as orchestrator_pb2
import tensorflow_elastic.orchestrator_pb2_grpc as orchestrator_pb2_grpc
import threading
NODE_GATHER_TIMEOUT = 30
HEALTH_CHECK_TIMEOUT = 10

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
    self._gathering = False
    self._state = "init"
    self._end_gathering = None
    self._param_lock = threading.Lock()
    self._wait_lock = threading.Lock()
    self._waiting_nodes = 0
    self._sync_lock = threading.Lock()
    self._bar_lock = threading.Lock()
    self._data_sync = {}
    self._data_bar = {}
    self._start_sync = {}
    self._start_bar = {}
    self._finish_bar = False
    self._finish_sync = False
    self._stop_event = threading.Event()
    
  def _create_cluster_spec(self):
    cluser_spec_template = {"cluster":{"worker":[]}, "task":{"index": None, "type":"worker"}}
    cluser_spec_template["cluster"]["worker"] = list(self._workers[self._current_id].keys())
    ret = json.dumps(cluser_spec_template)
    logging.info(ret)
    logging.info(self._workers)
    return ret


  def _update_state(self, reset=None, done=None):
    logging.warning("Trying to update state")
    self._lock.acquire()
    logging.warning("Acquired lock")
    if(self._state == "init"):
      self._state = "gather"

    # if(self._state == "run" and reset):
    #   logging.warning(f"Swapping state {self._state} to gather")
    #   self._state = "gather"
    #   #Reset the time to gather
    #   self._end_gathering = time.time() + NODE_GATHER_TIMEOUT
    #   logging.warning(f"Reset gathering time to {self._end_gathering}")
    #   self._workers = []
    #   self._waiting_nodes = 0

    if(self._state == "gather" and done):
      self._state = "run"

    self._lock.release()
    logging.warning("Released lock")

  def _add_to_wait(self):
    self._wait_lock.acquire()
    #Should only add yourself to waiting nodes if there is space in the cluster
    if(self._waiting_nodes == 0):
      self._next_id = self._current_id +1
    self._waiting_nodes += 1
    self._wait_lock.release()

    while(self._waiting_nodes < self._params["min_nodes"]):
      logging.warning("waiting")
      time.sleep(1)
    self._state = "gather"
    
    
    logging.warning("Stopping wait because we have reached min nodes")
    
    if(self._current_id != self._next_id):
      self._wait_lock.acquire()
      self._current_id = self._next_id
      if(self._current_id not in self._workers):
        #self._update_state(reset=True) #Reset once we get a new id
        self._state = "gather"
        self._workers[self._current_id] = {}
        #Reset End time
        self._end_gathering = time.time() + NODE_GATHER_TIMEOUT
      self._wait_lock.release()
    
      
    logging.warning(f"{self._min_barrier} {self._min_barrier.n_waiting} {self._min_barrier.parties}")
    self._min_barrier.wait()

    #logging.warning(f"Cur id {self._current_id}, next id {self._next_id}, workers {self._workers}")
    return
    
  def _verify_params(self, request):
    if(self._params is None):
      self._param_lock.acquire()
      if(self._params is None):
        self._params = {"min_nodes":request.min_nodes, "max_nodes":request.max_nodes}
        self._min_barrier = threading.Barrier(request.min_nodes)
      self._param_lock.release()
    
    #Verify params are the same
    v1 = self._params["min_nodes"] == request.min_nodes
    v2 = self._params["max_nodes"] == request.max_nodes
    return v1 and v2



    
  def GetClusterSpec(self, request, context):
    
    if(not self._verify_params(request)):
      logging.warning(f"Request {request} doesn't match params {self._params}")
      return orchestrator_pb2.ClusterSpec(cluster_spec="")
    logging.warning(f"{threading.get_ident()} : {request}")
    self._update_state()
    #We are already running, and a reset hasn't been requested
    # Reset requests happen when a node identifies num of waiting nodes is > 0    
    logging.warning(f"{threading.get_ident()} : {self._state}")
    if(self._state == "run"):
      logging.warning(f"State is run")
      self._add_to_wait()
      self._waiting_nodes = 0
    else:
      self._end_gathering = time.time() + NODE_GATHER_TIMEOUT
    
    #Currently we can't handle excess nodes
    self._lock.acquire()
    logging.warning(f"{threading.get_ident()} Acquired lock + {request}")
    self._workers[self._current_id][request.address] = time.time()
    self._lock.release()
    logging.warning("Released lock")

    
    
    while(len(self._workers[self._current_id]) < request.min_nodes):
      logging.warning(f"# of workers {len(self._workers[self._current_id])} hasn't reached Min nodes of {request.min_nodes}")
      time.sleep(1)
      

    logging.warning(f"Waiting to end gathering at {self._end_gathering}")
    while (time.time() < self._end_gathering ):
      if(len(self._workers[self._current_id]) < request.max_nodes):
        logging.info(f"Waiting for more nodes to gather, current node count {len(self._workers[self._current_id])}")
        time.sleep(NODE_GATHER_TIMEOUT*.1)
      else:
        logging.info(f"Max nodes reached {len(self._workers[self._current_id])}")
        #Head back to waiting?
        break
    else:
      logging.info("Max waiting time reached will continue")
      #Because the gather time is longer than Health check, prior to leaving, lets reset the health check
      self._workers[self._current_id][request.address] = time.time()

    self._update_state(done=True)
    #Create cluster spec from workers
    logging.info("Returning")
    return orchestrator_pb2.ClusterSpec(cluster_spec=self._create_cluster_spec())


  def GetWaitingNodes(self, request, context):
    # We will use this as a health check.
    # Healthy nodes should be checking this regularly. If a node fails to check this in HEALTH_CHECK_TIMEOUT
    # We assume this node is dead
    
    #First verify we are in the list of workers
    #logging.info(f"Getting waiting nodes, {request.address} : {self._workers[self._current_id]}")
    address = request.address
    if(address not in self._workers[self._current_id]):
      #Return an error message if we fail
      logging.info("Returning failed")
      return orchestrator_pb2.WaitingNodes(num_waiting_nodes=self._waiting_nodes, error_msg=f"{address} not in current worker {self._workers[self._current_id]}") 
    
    self._workers[self._current_id][address] = time.time()

    if(self._waiting_nodes > 0):
      return orchestrator_pb2.WaitingNodes(num_waiting_nodes=self._waiting_nodes, error_msg="")
    
    #If we are already waiting for nodes (being added) we can return
    #now check for Health check failures

    fail = [self._workers[self._current_id][x] + HEALTH_CHECK_TIMEOUT > time.time() for x in self._workers[self._current_id]]
    num_failed = 0-fail.count(False)
    #logging.info(f"Number of failed {num_failed}, Workers {self._workers[self._current_id]}")

    return orchestrator_pb2.WaitingNodes(num_waiting_nodes=num_failed, error_msg="")  


  def Synchronize(self, request, context):
    # Blocking call/timeout that will synchronize data, and dump out as json string
    def ret_val(ret_bool: bool, data: str, msg=""):
      self._start_sync = False
      return orchestrator_pb2.SyncResponse (success=ret_bool, data=data, error_msg=msg)

    end_time = time.time() + (2^62-1) if(request.timeout <= 0) else time.time() + request.timeout

    
    self._sync_lock.acquire()
    if(not self._start_sync):
      self._data_sync = {}
      self._start_sync = True
    
    if(request.address not in self._workers[self._current_id]):
      self._sync_lock.release()
      return ret_val(False, "{}", f"{request.address} not in current list of workers {str(self._workers)}")

    self._data_sync[request.address] = request.data
    self._sync_lock.release()

    while (time.time() <= end_time):
      sync_len = len(self._data_sync)
      if(sync_len == len(self._workers[self._current_id])):
        return ret_val(True, json.dumps(self._data_sync))
      elif(sync_len > len(self._workers[self._current_id])):
        self._start_sync = False
        raise ValueError(f"{sync_len} num workers synchronizing > {len(self._workers[self._current_id])} ")
      else:
        time.sleep(1)
        logging.warning(f"{sync_len}")
    if(len(self._data_sync) == len(self._workers[self._current_id])):
        return ret_val(True, json.dumps(self._data_sync))
    else:
      self._data_sync.pop(request.address, None)
      return ret_val(False, "{}", f"Timeout reached {request.timeout}")

  def Barrier(self, request, context):
    def ret_val(ret_bool: bool, msg=""):
      self._start_bar = False
      return orchestrator_pb2.BarrierResponse (success=ret_bool, error_msg=msg)

    end_time = time.time() + (2^62-1) if(request.timeout <= 0) else time.time() + request.timeout

    self._bar_lock.acquire()
    if(not self._start_bar):
      self._data_bar = {}
      self._start_bar = True
    
    if(request.address not in self._workers[self._current_id]):
      self._bar_lock.release()
      return ret_val(False, f"{request.address} not in current list of workers {str(self._workers[self._current_id])}")

    self._data_bar[request.address]=1
    self._bar_lock.release()
    
    while (time.time() <= end_time):
      bar_len = len(self._data_bar)
      if(bar_len == len(self._workers[self._current_id])):
        return ret_val(True, "")
      elif(bar_len > len(self._workers[self._current_id])):
        self._start_bar = False
        raise ValueError(f"{bar_len} num workers synchronizing > {len(self._workers[self._current_id])} ")
      else:
        time.sleep(1)
    if(len(self._data_bar) == len(self._workers[self._current_id])):
        return ret_val(True, "")
    else:
      self._data_bar.pop(request.address, None)
      return ret_val(False, f"Timeout reached {request.timeout}")

  

def serve(port="50051"):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    orchestrator_pb2_grpc.add_TFEOrchestratorServicer_to_server(
        TFEOrchestratorServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()
    


if __name__ == '__main__':
    logging.basicConfig()
    serve()

