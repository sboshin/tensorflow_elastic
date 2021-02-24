import os
import utils
import time
from multiprocessing import Pool, Process
from multiprocessing import TimeoutError as mpTimeoutError

#Idea is a class to create

def run_script(hostname, key, tfconfig, script, args, local, log_dir,private_ip, add_python=True):
  with utils.setup_env(hostname, key) as connection:
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


    if(add_python):
      connection.run(f"python3.6 {remote_script} {args} |& tee -a {worker_log}", env=myenv)
      connection.get(worker_log, f"{log_dir}/{private_ip}_eworker.log")
    else:
      connection.run(f"{remote_script} {args}")
      

def run_script2(hostname, key, tfconfig, script, args, local, log_dir, private_ip, use_python=True, run_prefix=""):
  with utils.setup_env(hostname, key) as connection:
    basename = os.path.basename(script)
    if(not local):
      remote_script = f"/home/ubuntu/{basename}"
      connection.put(script, remote_script)
    else:
      remote_script = script
    worker_log = f"/home/ubuntu/worker.log"
    
    #Add run prefix
    remote_script = run_prefix+remote_script

    myenv  = {}
    myenv.update(utils.CUSTOMENV)
    myenv["TF_CONFIG"] = f"'{tfconfig}'"

    print(f"use_python {use_python}", flush=True)
    if(use_python):
      print(f"python3.6 {remote_script} {args} |& tee {worker_log}")
      connection.run(f"python3.6 {remote_script} {args} |& tee {worker_log}", env=myenv)
    else:
      connection.run(f"{remote_script} {args} |& tee {worker_log}", env=myenv)
    connection.get(worker_log, f"{log_dir}/{private_ip}_worker.log")
  

class ElasticCluster():
  def __init__(self, instance_type, instance_cnt, ami, key, setup_script, elastic_pattern, training_script_params, local, orchestrator, log_dir="./", run_prefix=""):
    self._instance_type = instance_type
    self._instance_cnt = instance_cnt
    self._ami = ami
    self._key = key
    self._setup_script = setup_script
    self._elastic_pattern = elastic_pattern
    self._training_script_params = training_script_params
    self._local = local
    self._log_dir = log_dir
    self._orchestrator = orchestrator
    self._run_prefix = run_prefix

    self.setup_instances()


  def setup_instances(self):
    #Start all instances, and run setup scripts
    self._instance_ids, self._hostnames, self._private_ips = utils.start_instances(self._instance_type, self._instance_cnt, self._ami, self._key, "elastic_worker")

    #Run setup script
    utils.run_setup_script_on_hostnames(self._setup_script, self._hostnames, self._key, self._instance_ids)

    #Instances should be ready

  def terminate_scripts_on_hosts(self):
    pool = Pool(processes=len(self._hostnames))
    mp_args = []
    print(f"Terminating hosts on {self._hostnames}")
    for hostname in self._hostnames:
      mp_args.append((hostname, self._key, {}, "killall python3.6", "", True, self._log_dir, hostname, False))
    try:
      pool.starmap(run_script, mp_args)
    except Exception as e:
      print(e)


  def run_pattern(self):
    pattern, pattern_time = self._elastic_pattern.split(":")
    pattern_time = float(pattern_time) * 60 #Convert to seconds
    #Cycle pattern, all run then all stop
    if(pattern == "cycle"):
      print(f"Running Cycle pattern with {pattern_time} pattern time")
      #Start with all off
      time.sleep(pattern_time)
      
      while(True):
        pool = Pool(processes=len(self._hostnames))
        #Setup Run
        hostname_clusterspec = utils.setup_tf_config(self._hostnames, self._private_ips, 5555,self._orchestrator)
        mp_args = []
        for hostname in hostname_clusterspec:
          private_ip = self._private_ips[self._hostnames.index(hostname)]
          mp_args.append((hostname, self._key, hostname_clusterspec[hostname], 
            self._training_script_params["script"], self._training_script_params["args"], self._local, self._log_dir, private_ip, True, self._run_prefix))

        async_result = pool.starmap_async(run_script2, mp_args)
        try:
          final_result = async_result.get(float(pattern_time))
          print("Call finished will return")
          utils.terminate_instances(self._instance_ids)
          break
        except mpTimeoutError:
          #Timeout waited enough with no result
          print(f"Timeout Reached, refreshing instance after {pattern_time}s", flush=True)
          self.terminate_scripts_on_hosts()
          pool.terminate()
          pool.join()
          time.sleep(pattern_time)
        except Exception as e:
          print(e)
          #utils.terminate_instances(self._instance_ids)
          break
    try:
      utils.terminate_instances(self._instance_ids)
    except Exception:
      pass
    
        
          

          

  def start(self):
    self._ctrlpid = Process(target=self.run_pattern)
    self._ctrlpid.start()
    pass

  def stop(self):
    try:
      utils.terminate_instances(self._instance_ids)
    except Exception as e:
      print(e)
      pass
    self._ctrlpid.terminate()
    self._ctrlpid.join()


if __name__ == "__main__":
  pass  