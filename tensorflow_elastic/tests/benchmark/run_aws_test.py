import os
import sys
import boto3
import botocore
import time
from argparse import REMAINDER, ArgumentParser
import atexit
import paramiko
import copy
import json
import socket
import shutil
from multiprocessing import Pool, Process
from elastic_cluster import ElasticCluster
from fabric.connection import Connection
from utils import start_instances, terminate_instances, setup_env, DEFAULT_REGION, run_setup_script_on_hostnames, setup_tf_config
import utils


#socket.setdefaulttimeout(10)

#This will be a framework to run benchmarks and also setting up aws resources
#Geared towards MWMS (Multi-worker mirrored strategy)
#Specify number of instances, instance type, script and script args. And it will start resources, setup, and should return, and kill on error, with option of not killing

def parse_args(args):
  parser = ArgumentParser(description="MWMS aws runner")
  parser.add_argument(
    "--instance_type", type=str, help="Instance type to run on aws"
  )
  parser.add_argument(
    "--instance_cnt", type=int, help="Number of Instances to run on aws"
  )
  parser.add_argument(
    "--elastic", type=bool, help="Run with tensorflow elastic", default=False
  )
  parser.add_argument(
    "--ami", type=str, help="AMI ID"
  )
  parser.add_argument(
    "--setup_script", type=str, help="Any setup instructions, Assumption is BASE ami and no virtual env or conda usage", default=""
  )
  parser.add_argument(
    "--key_name", type=str, help="Any setup instructions, Assumption is BASE ami and no virtual env or conda usage"
  )
  parser.add_argument(
    "--local", default=False, action="store_true", help="Treat the training script as a local script on the AMI"
  )
  parser.add_argument(
    "--elastic_nodes", type=int, default=0 , help="Number of nodes that should behave like the elastic pattern"
  )
  parser.add_argument(
    "--elastic_pattern", type=str, default="cycle:7", help="Dynamic pattern to follow for elastic nodes (Random|step|cycle)"
  )

  parser.add_argument(
    "--log_dir", type=str, default="./", help="Directory to dump out logs"
  )

  #positional
  parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script",
  )

  # rest from the training program
  parser.add_argument("training_script_args", nargs=REMAINDER)
  return parser.parse_args(args)

def run_script(hostname, key, tfconfig, script, args, local, log_dir, private_ip):
  with setup_env(hostname, key) as connection:
    basename = os.path.basename(script)
    if(not local):
      remote_script = f"/home/ubuntu/{basename}"
      connection.put(script, remote_script)
    else:
      remote_script = script
    worker_log = f"/home/ubuntu/worker.log"
    
    myenv  = {}
    myenv.update(utils.CUSTOMENV)
    myenv["TF_CONFIG"] = f"'{tfconfig}'"

    connection.run(f"python {remote_script} {args} |& tee {worker_log}", env=myenv)
    connection.get(worker_log, f"{log_dir}/{private_ip}_worker.log")

def main(args=None):
  # If ``args`` not passed, defaults to ``sys.argv[:1]``
  args = parse_args(args)

  #Validate log_dir
  if(os.path.isdir(args.log_dir)):
    shutil.rmtree(args.log_dir)
  if(not os.path.isdir(args.log_dir)):
    os.mkdir(args.log_dir)
  


  #start instances
  instance_ids, hostnames, private_ips = start_instances(args.instance_type, args.instance_cnt, args.ami, args.key_name,"worker")
  atexit.register(terminate_instances, (instance_ids))
  
  if(args.setup_script != ""):
    run_setup_script_on_hostnames(args.setup_script, hostnames, args.key_name, instance_ids)

    #run_setup_script(args.setup_script, hostname, args.key_name)

  #Setup TFCONFIG
  hostname_clusterspec = setup_tf_config(hostnames, private_ips, 5555)

  training_script = args.training_script
  training_script_args = " ".join(args.training_script_args)

  pool = Pool(processes=args.instance_cnt)
  #If elastic nodes setup elastic
  if(args.elastic_nodes != 0):
    #Start elastic server on c5.large
    orchestrator_type = "c5.large"
    e_instance_id, e_hostname, e_private_ip = start_instances(orchestrator_type, 1, args.ami, args.key_name, "orchestrator")
    instance_ids += e_instance_id
    if(args.setup_script != ""):
      #we should have a standard setup script for orchestrators
      run_setup_script_on_hostnames(args.setup_script, e_hostname, args.key_name, instance_ids)
    
    #Start orchestrator
    orch_training_script = "-m tensorflow_elastic.distributed.launch --standalone --rdzv_endpoint=localhost:5556 start"
    orch_args = (e_hostname[0], args.key_name, "{}", orch_training_script, "", True, args.log_dir, e_private_ip[0])
    orch_proc = Process(target=run_script, args=orch_args)
    

    training_script = f"-m tensorflow_elastic.distributed.launch --monitor_interval=5 --address=`hostname -i`:5555 --nnodes={args.instance_cnt}:{args.instance_cnt+args.elastic_nodes} --rdzv_endpoint={e_private_ip[0]}:5556 {training_script}"
    
    training_params = {"script":training_script, "args":training_script_args}
    #Setup Elastic Cluster
    e_cluster = ElasticCluster(args.instance_type, args.elastic_nodes, args.ami, args.key_name, 
      args.setup_script, args.elastic_pattern, training_params, True, args.log_dir)

    #start orchestrator
    orch_proc.start()
  
  mp_args = []
  for hostname in hostname_clusterspec:
    private_ip = private_ips[hostnames.index(hostname)]
    mp_args.append((hostname, args.key_name, hostname_clusterspec[hostname], training_script, training_script_args, args.local, args.log_dir, private_ip))
    
  try:
    async_result = pool.starmap_async(run_script, mp_args)
    if(args.elastic_nodes != 0):
      e_cluster.start()
    async_result.get()

  finally:
    if(args.elastic_nodes != 0):
      e_cluster.stop()
      orch_proc.terminate()
      orch_proc.join()
      with setup_env(e_hostname[0], args.key_name) as connection:
        worker_log = f"/home/ubuntu/worker.log"
        try:
          connection.get(worker_log, f"{args.log_dir}/{e_private_ip[0]}_worker.log")
        except Exception as e:
          print(e)
    terminate_instances(instance_ids)
    


if __name__ == "__main__":
  main()