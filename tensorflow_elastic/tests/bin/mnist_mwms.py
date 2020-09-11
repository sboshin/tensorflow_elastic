import os
import argparse
import tensorflow as tf
import time
import json
import numpy as np
from tensorflow.python.keras.distribute.worker_training_state import WorkerTrainingState
from tensorflow.python.lib.io import file_io

def delete_backup(self):
  """Delete the backup directories.
  Delete the backup directories which should not exist after `fit()`
  successfully finishes.
  """
  # pylint: disable=protected-access

  try:
    for pathname in file_io.get_matching_files(
        self.write_checkpoint_manager._prefix + '*'):
      file_io.delete_recursively(pathname)
  except tf.python.framework.errors_impl.NotFoundError as e:
    print(e, flush=True)
  try:
    for pathname in file_io.get_matching_files(
        os.path.join(self.write_checkpoint_manager.directory, 'checkpoint')):
      file_io.delete_recursively(pathname)
  except tf.python.framework.errors_impl.NotFoundError as e:
    print(e, flush=True)
    

WorkerTrainingState.delete_backup = delete_backup

from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceStrategy

def parse_args():
    parser = argparse.ArgumentParser(description="tf_config test")

    parser.add_argument(
      "--epochs", type=int, help="sleep time inbetween iterations", default=15
    )
    return parser.parse_args()

def mnist_dataset(batch_size):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the range [0, 255].
  # We need to convert them to float32 with values in the range [0, 1]
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
  return train_dataset


def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model


def main():
    """
    The tests should be run assuming we start our process with 
    """
    args = parse_args()
    print(os.environ)
    assert "TF_CONFIG" in os.environ

    tf_config = os.environ["TF_CONFIG"]
    print("TF_CONFIG env is %s "%(tf_config), flush=True)
    tf_config = json.loads(tf_config)
    
    strategy = CollectiveAllReduceStrategy()
    num_workers = strategy.extended._cluster_resolver.cluster_spec().num_tasks("worker")
    per_worker_batch_size = 64

    tmp_dir = "/tmp/backup"
    

    # Here the batch size scales up by number of workers since 
    # `tf.data.Dataset.batch` expects the global batch size. Previously we used 64, 
    # and now this becomes 128.
    global_batch_size = per_worker_batch_size * num_workers
    multi_worker_dataset = mnist_dataset(global_batch_size)

    with strategy.scope():
      # Model building/compiling need to be within `strategy.scope()`.
      multi_worker_model = build_and_compile_cnn_model()

    # Keras' `model.fit()` trains the model with specified number of epochs and
    # number of steps per epoch. Note that the numbers here are for demonstration
    # purposes only and may not sufficiently produce a model with good quality.
    callbacks = [tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=tmp_dir)]

    multi_worker_model.fit(multi_worker_dataset, epochs=30, steps_per_epoch=args.epochs, callbacks=callbacks)

    return 0
    

    

if __name__ == "__main__":
    main()

