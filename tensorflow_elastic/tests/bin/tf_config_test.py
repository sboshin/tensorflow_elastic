import os
import argparse
import tensorflow
import time
import tensorflow_elastic.tensorflow

from tensorflow.python.distribute.cluster_resolver.tfconfig_cluster_resolver import TFConfigClusterResolver

def parse_args():
    parser = argparse.ArgumentParser(description="tf_config test")

    parser.add_argument(
      "--sleep", type=int, help="sleep time inbetween iterations", default=15
    )


    return parser.parse_args()

def main():
    """
    The tests should be run assuming we start our process with 
    """
    args = parse_args()

    assert "TF_CONFIG" in os.environ

    tf_config = os.environ["TF_CONFIG"]
    print("TF_CONFIG env is %s "%(tf_config), flush=True)
    
    cluster_resolver = TFConfigClusterResolver()
    #Cluster resolver commands to validate proper tfconfig
    cluster_spec = cluster_resolver.cluster_spec()
    
    assert cluster_resolver.task_id is not None
    assert cluster_resolver.task_type is not None
    assert cluster_spec is not None
    #assert int(cluster_resolver.task_id) == int(os.environ["RANK"]), f"task_id {cluster_resolver.task_id} | rank {os.environ['RANK']}"

    print(f"Task Id {cluster_resolver.task_id} and task type {cluster_resolver.task_type}", flush=True)
    
    #As long as failures happen we should restart before sleep is over
    time.sleep(args.sleep)

    

if __name__ == "__main__":
    main()

