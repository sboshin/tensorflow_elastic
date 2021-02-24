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
from multiprocessing import Pool

from fabric.connection import Connection

DEFAULT_REGION='us-west-2'

CUSTOMENV = {}
CUSTOMENV["LD_LIBRARY_PATH"] = "/usr/local/cuda-10.1/lib64"

#Can we generalize this?
CUSTOMENV["AWS_REGION"]="us-west-2"
CUSTOMENV["S3_ENDPOINT"]="s3.us-west-2.amazonaws.com"
CUSTOMENV["PYTHONPATH"]="/home/ubuntu/src/cntk/bindings/python:/home/ubuntu/benchmark/models"
CUSTOMENV["TF_CPP_MIN_LOG_LEVEL"] = 0
CUSTOMENV["TF_CPP_MIN_VLOG_LEVEL"] = 0
CUSTOMENV["TF_DETERMINISTIC_OPS"] = 1
CUSTOMENV["LD_LIBRARY_PATH"] = "/usr/local/cuda-11.0/extras/CUPTI/lib64:/usr/lib64/openmpi/lib/:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/mpi/lib:/lib/:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/usr/local/cuda/lib:/opt/amazon/efa/lib:/usr/local/mpi/lib:/usr/lib64/openmpi/lib/:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/mpi/lib:/lib/:"


def delete_s3_key(bucket, key, region):
  s3 = boto3.resource('s3')
  bucket = s3.Bucket(bucket)
  bucket.objects.filter(Prefix=f"{key}/").delete()
  

def setup_tf_config(hostnames, private_ips, port, orchestrator=""):
  ports_added = [f"{ip}:{port}" for ip in private_ips]
  cluster_temp = {"cluster":{"worker":ports_added, "orchestrator":[orchestrator]}, "task":{"index":None, "type":"worker"}}
  cluster_spec_hostname = {}
  for ii, _ in enumerate(private_ips):
    tmp_cspec = copy.copy(cluster_temp)
    tmp_cspec["task"]["index"] = ii
    
    cluster_spec_hostname[hostnames[ii]] = json.dumps(tmp_cspec)

  return cluster_spec_hostname

def run_setup_script_on_hostnames(setup_script, hostnames, key, instance_ids):
  print(f"\nSetting up {hostnames}\n", flush=True)
  mp_args = []
  for hostname in hostnames:
    mp_args.append((setup_script, hostname, key))
  setup_pool = Pool(processes=len(hostnames))
  try:
    setup_pool.starmap(run_setup_script, mp_args)
  except Exception as e:
    print(e)
    #terminate_instances(instance_ids)
    raise e
  print(f"\nFinished setting up {hostnames}\n", flush=True)

def run_setup_script(setup_script, hostname, key):
  #Copy setup script and run
  with setup_env(hostname, key) as connection:
    basename = os.path.basename(setup_script)
    remote_setup_script = f"/home/ubuntu/{basename}"
    connection.put(setup_script, remote_setup_script)
    connection.run(f"chmod +x {remote_setup_script}")
    connection.run(f"{remote_setup_script}")

def start_instances(instance_type, instance_cnt, ami, key, suffix=""):
  client = boto3.client('ec2', region_name=DEFAULT_REGION)
  tagspec = {"ResourceType":"instance", "Tags":[{"Key":"Name", "Value":f"run-aws-test-{suffix}"}]}
  response = client.run_instances(
    InstanceType=instance_type,
    MaxCount=instance_cnt,
    MinCount=instance_cnt,
    ImageId=ami,
    KeyName=key,
    SecurityGroups=["tfmwms"],
    TagSpecifications=[tagspec],
    IamInstanceProfile={"Name":"sboshin_eia"}
  )
  instance_ids = [instance["InstanceId"] for instance in response["Instances"]]

  #Get instance hostname
  hostnames = []
  private_ips = []
  while(True):
    try:
      response = client.describe_instances(InstanceIds=instance_ids)
      hostnames = [instance["PublicDnsName"] for reservation in response["Reservations"] for instance in reservation["Instances"]]
      private_ips = [instance["PrivateIpAddress"] for reservation in response["Reservations"] for instance in reservation["Instances"]]
      if("" not in hostnames):
        break
      else:
        #print(f"Can't find all hostnames {hostnames}")
        time.sleep(5)
    except botocore.exceptions.ClientError:
      time.sleep(5)

  #try until you can connect to ssh
  while(True):
    try:
      for hostname in hostnames:
        start = time.time()
        with setup_env(hostname, key) as connection:
          connection.run("ls", hide=True)
      break
    except paramiko.ssh_exception.NoValidConnectionsError as e:
      print(e)
      time.sleep(10)
    except (socket.timeout, TimeoutError) as e:
      print(e)
      print(f"Timeout took {time.time() - start}")
      time.sleep(10)
      
  print(f"\nFinished provisioning {instance_cnt} {instance_type} instance(s)\n")
  print(f"{instance_ids}, {hostnames}, {private_ips}")
  return instance_ids, hostnames, private_ips

def terminate_instances(instance_ids):
    client = boto3.client("ec2", region_name=DEFAULT_REGION)
    _ = client.terminate_instances(
    InstanceIds=instance_ids,
    )

def setup_env(hostname, key):
  
  connect = Connection(
    host=hostname,
    user="ubuntu",
    connect_kwargs={
        "key_filename": f"{key}.pem",
    },
    #connect_timeout=(2**30),
    inline_ssh_env=True,
  )
  return connect


if __name__ == "__main__":
  pass