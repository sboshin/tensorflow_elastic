import os
import argparse
import tensorflow as tf
import time
import tensorflow_elastic.tensorflow

from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceStrategy

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
    print("TF_CONFIG env is %s "%(tf_config))
    
    strategy = CollectiveAllReduceStrategy()
    strategy.extended._check_health_interval = 5
    with strategy.scope():
      var = strategy.reduce("SUM", tf.convert_to_tensor(1.), axis=None)

    print(f"Variable val: {var.numpy()} num workers {strategy.extended._num_workers}", flush=True)
    assert var.numpy() == strategy.extended._num_workers

    
    #As long as failures happen we should restart before sleep is over
    time.sleep(args.sleep)

    

if __name__ == "__main__":
    main()

