#!/usr/bin/env python3

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import time
import tensorflow_elastic.rendezvous.orchestrator_api as orch_api
import tensorflow as tf
import traceback

handler = None
FIRST_RUN = True
NUM_RETRIES = 3
MY_ADDRESS = None
cur_dataset = None


def get_handler(minN, maxN):
    tf_config = json.loads(os.environ["TF_CONFIG"])
    if("orchestrator" not in tf_config["cluster"]):
      raise ValueError("No Orchestrator in TFCONFIG %s"%(tf_config))
  
    if(not isinstance(tf_config["cluster"]["orchestrator"], list)):
      raise ValueError("Orchestrator isn't defined as a list %s"%(tf_config["cluster"]["orchestrator"]))

    server_address = tf_config["cluster"]["orchestrator"][0]
    return orch_api.TFEOrchestratorHandler(server_address, minN, maxN)

def set_tfconfig(tf_config):
    os.environ["TF_CONFIG"] = json.dumps(tf_config)

def init(my_address, min_nodes, max_nodes, orchestrator_address="", num_retries=3):
  global handler, NUM_RETRIES, MY_ADDRESS
  #Get handler
  NUM_RETRIES = num_retries
  MY_ADDRESS = my_address
  handler = get_handler(min_nodes, max_nodes)
  cluster_spec = {"cluster":{"worker":{}}, "task":{"index": None, "type":"worker"}}
  #Get Unique index for my address
  unique_index = handler.UniqueIndex(my_address)

  #Setup initial tf_config
  cluster_spec["cluster"]["worker"] = {unique_index:my_address}
  cluster_spec["task"]["index"] = unique_index
  set_tfconfig(cluster_spec)
  return
    
def join_cluster(strategy):
  cluster_spec = handler.GetClusterSpec(MY_ADDRESS, False)
  set_tfconfig(cluster_spec)
  strategy.extended.update_cluster()
  handler.Barrier(MY_ADDRESS, "getspec", 180)
  if strategy.extended._enable_check_health:
    #if(strategy.extended._num_workers > 1):
    strategy.extended._start_check_health_thread()
    
def getIterKwargs(strategy, dataset):
  s2 = time.time()
  dataset_dist = strategy.experimental_distribute_dataset(dataset)
  s3 = time.time()
  dataset_iter = iter(dataset_dist)
  print(f"dist dataset {s3-s2} iter data {time.time() - s3}")
  return {"iterator":dataset_iter}

def barrier(tag, timeout):
  handler.Barrier(MY_ADDRESS, tag, timeout)

def run(run_fn, run_fn_args, dataset_fn, strategy, checkpoint_manager):
  run_start = time.time()
  global FIRST_RUN
  global cur_dataset
  
  if(FIRST_RUN):
    #Warmup, and Join cluster
    #Warmup is running the model and resetting the variables
    run_kwargs = getIterKwargs(strategy, dataset_fn())
    run_fn(*run_fn_args, **run_kwargs) #we are losing one step here
    join_cluster(strategy)
    FIRST_RUN = False
    print(f"Restoring from {checkpoint_manager.directory}")
    checkpoint_manager.checkpoint.restore(checkpoint_manager.directory)
    print(f"First Run took {time.time() - run_start}")
    cur_dataset = dataset_fn()

  after_first = time.time()
  #Check to see if we have any waiting nodes
  waiting = handler.GetWaitingNodes(MY_ADDRESS)
  if(waiting > 0): #Grown
    print(f"Growing Cluster", flush=True)
    strategy.extended._stop_check_health_thread()
    join_cluster(strategy)
    cur_dataset = dataset_fn()
    checkpoint_manager.checkpoint.restore(checkpoint_manager.directory)
    print(f"Growing Cluster took {time.time() - after_first}")

  cur_try = 0
  failure_time = 0
  epoch_lost_time = 0
  
  
  run_kwargs = getIterKwargs(strategy, cur_dataset)

  actual_run_start = time.time()
  true_epoch_start = time.time()

  steps = 0
  while(True):
    try:
      print(f"Starting train Step", flush=True)
      start_step = time.time()
      run_fn(*run_fn_args, **run_kwargs)
      steps +=1
      print(f"Current Step {steps} took {time.time() - start_step} and TF_CONFIG is {os.environ['TF_CONFIG']}", flush=True)
    except (tf.errors.OutOfRangeError, StopIteration) as e:
      end = time.time()
      print(f"Finished Run: total time {end - actual_run_start}s failure time {failure_time}s time without failure {end - actual_run_start - failure_time}s")
      print(f"True Epoch time {end - true_epoch_start}s epoch time lost {epoch_lost_time}s")
      return True
    except Exception as e:
      #traceback.print_stack()
      print("\n\n\n\n\n\n\n\n\n\n",flush=True)
      print(f"{e}")
      print("\n\n\n\n\n\n\n\n\n\n",flush=True)

      #raise e
      failure_start = time.time()
      epoch_lost_time += failure_start - true_epoch_start
      cur_try +=1
      steps = 0
      print(f"Current error is {e}\n\n", flush=True)
      
      strategy.extended._stop_check_health_thread()
      join_cluster(strategy)
      checkpoint_manager.checkpoint.restore(checkpoint_manager.directory)
      cur_dataset = dataset_fn()
      run_kwargs = getIterKwargs(strategy, cur_dataset)
      failure_time += time.time() - failure_start
      if cur_try > NUM_RETRIES:
       raise e
      true_epoch_start = time.time()
      




