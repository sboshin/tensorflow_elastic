import os
import argparse
import shutil
import tensorflow as tf
import time
import json
import numpy as np
import socket
from tensorflow.python.keras.distribute.worker_training_state import WorkerTrainingState
from tensorflow.python.lib.io import file_io
from absl import logging
import tensorflow_elastic as tfe
import tensorflow_elastic.rendezvous.orchestrator_api as orch_api

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
      "--epochs", type=int, help="sleep time inbetween iterations", default=5
    )
    parser.add_argument(
      "--steps", type=int, help="Number of steps in dataset (0 will ignore take ds operation)", default=2
    )
    parser.add_argument(
      "--index", type=int, help="Number of steps in dataset (0 will ignore take ds operation)", default=0
    )
    return parser.parse_args()

def cifar_dataset(batch_size, steps):
  # (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # # The `x` arrays are in uint8 and have values in the range [0, 255].
  # # We need to convert them to float32 with values in the range [0, 1]
  # x_train = x_train / np.float32(255)
  # y_train = y_train.astype(np.int64)
  # train_dataset = tf.data.Dataset.from_tensor_slices(
  #     (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)

  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

  # Normalize pixel values to be between 0 and 1
  train_images, test_images = train_images / 255.0, test_images / 255.0

  def add_options(images, labels):
    dataset = tf.data.Dataset.from_tensor_slices(
        (images, labels)).shuffle(60000).batch(batch_size)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)
    return dataset

  return add_options(train_images, train_labels), add_options(test_images, test_labels)


def build_and_compile_cnn_model():
  model = tf.keras.applications.MobileNetV2(
    include_top=True, weights=None, input_tensor=None, input_shape=(32, 32, 3),
    pooling=True, classes=10, classifier_activation='softmax',
  )
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=optimizer,
      metrics=['accuracy'])
  return model, optimizer

def _run_callbacks(callbacks, grads_and_vars):
  for callback in callbacks:
    grads_and_vars = callback(grads_and_vars)
  return grads_and_vars

def _filter_grads(grads_and_vars):
  """Filter out iterable with grad equal to None."""
  grads_and_vars = tuple(grads_and_vars)
  if not grads_and_vars:
    return grads_and_vars
  filtered = []
  vars_with_empty_grads = []
  for grad, var in grads_and_vars:
    if grad is None:
      vars_with_empty_grads.append(var)
    else:
      filtered.append((grad, var))
  filtered = tuple(filtered)
  if not filtered:
    raise ValueError("No gradients provided for any variable: %s." %
                     ([v.name for _, v in grads_and_vars],))
  if vars_with_empty_grads:
    logging.warning(
        ("Gradients do not exist for variables %s when minimizing the loss."),
        ([v.name for v in vars_with_empty_grads]))
  return filtered

def _filter_and_allreduce_gradients(grads_and_vars,
                                    allreduce_precision="float32",
                                    bytes_per_pack=0):
  """Filter None grads and then allreduce gradients in specified precision.
  This utils function is used when users intent to explicitly allreduce
  gradients and customize gradients operations before and after allreduce.
  The allreduced gradients are then passed to optimizer.apply_gradients(
  experimental_aggregate_gradients=False).
  Arguments:
      grads_and_vars: gradients and variables pairs.
      allreduce_precision: Whether to allreduce gradients in float32 or float16.
      bytes_per_pack: A non-negative integer. Breaks collective operations into
        packs of certain size. If it's zero, all gradients are in one pack.
  Returns:
      pairs of allreduced non-None gradients and variables.
  """
  filtered_grads_and_vars = _filter_grads(grads_and_vars)
  (grads, variables) = zip(*filtered_grads_and_vars)
  if allreduce_precision == "float16":
    grads = [tf.cast(grad, "float16") for grad in grads]
  hints = tf.distribute.experimental.CollectiveHints(
      bytes_per_pack=bytes_per_pack)
  allreduced_grads = tf.distribute.get_replica_context().all_reduce(
      tf.distribute.ReduceOp.SUM, grads, hints)
  if allreduce_precision == "float16":
    allreduced_grads = [tf.cast(grad, "float32") for grad in allreduced_grads]
  return allreduced_grads, variables

def minimize_using_explicit_allreduce(tape,
                                      optimizer,
                                      loss,
                                      trainable_variables,
                                      pre_allreduce_callbacks=None,
                                      post_allreduce_callbacks=None,
                                      allreduce_bytes_per_pack=0):
  """Minimizes loss for one step by updating `trainable_variables`.
  Minimizes loss for one step by updating `trainable_variables`.
  This explicitly performs gradient allreduce, instead of relying on implicit
  allreduce in optimizer.apply_gradients(). If training using FP16 mixed
  precision, explicit allreduce will aggregate gradients in FP16 format.
  For TPU and GPU training using FP32, explicit allreduce will aggregate
  gradients in FP32 format.
  Arguments:
      tape: An instance of `tf.GradientTape`.
      optimizer: An instance of `tf.keras.optimizers.Optimizer`.
      loss: the loss tensor.
      trainable_variables: A list of model Variables.
      pre_allreduce_callbacks: A list of callback functions that takes gradients
        and model variables pairs as input, manipulate them, and returns a new
        gradients and model variables pairs. The callback functions will be
        invoked in the list order and before gradients are allreduced. With
        mixed precision training, the pre_allreduce_allbacks will be applied on
        scaled_gradients. Default is no callbacks.
      post_allreduce_callbacks: A list of callback functions that takes
        gradients and model variables pairs as input, manipulate them, and
        returns a new gradients and model variables paris. The callback
        functions will be invoked in the list order and right before gradients
        are applied to variables for updates. Default is no callbacks.
      allreduce_bytes_per_pack: A non-negative integer. Breaks collective
        operations into packs of certain size. If it's zero, all gradients are
        in one pack.
  """
  if isinstance(optimizer,
                tf.keras.mixed_precision.experimental.LossScaleOptimizer):
    # FP16 GPU code path
    with tape:
      scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_grads = tape.gradient(scaled_loss, trainable_variables)
    grads_and_vars = zip(scaled_grads, trainable_variables)
    if pre_allreduce_callbacks:
      grads_and_vars = _run_callbacks(pre_allreduce_callbacks, grads_and_vars)
    (allreduced_scaled_grads,
     filtered_training_vars) = _filter_and_allreduce_gradients(
         grads_and_vars,
         allreduce_precision="float16",
         bytes_per_pack=allreduce_bytes_per_pack)
    allreduced_unscaled_grads = optimizer.get_unscaled_gradients(
        allreduced_scaled_grads)
    grads_and_vars = zip(allreduced_unscaled_grads, filtered_training_vars)
  else:
    # TPU or FP32 GPU code path
    grads = tape.gradient(loss, trainable_variables)
    grads_and_vars = zip(grads, trainable_variables)
    if pre_allreduce_callbacks:
      grads_and_vars = _run_callbacks(pre_allreduce_callbacks, grads_and_vars)
    (allreduced_grads,
     filtered_training_vars) = _filter_and_allreduce_gradients(
         grads_and_vars,
         allreduce_precision="float32",
         bytes_per_pack=allreduce_bytes_per_pack)
    grads_and_vars = zip(allreduced_grads, filtered_training_vars)
  if post_allreduce_callbacks:
    grads_and_vars = _run_callbacks(post_allreduce_callbacks, grads_and_vars)
  optimizer.apply_gradients(
      grads_and_vars, experimental_aggregate_gradients=False)

def train_step(model, strategy, batchsize, optimizer, iterator):
    """See base class."""

    @tf.function
    def step_fn(inputs):
      """Function to run on the device."""
      images, labels = inputs
      with tf.GradientTape() as tape:
        logits = model(images, training=True)

        prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits)
        loss = tf.reduce_sum(prediction_loss) * (1.0 / batchsize)
        num_replicas = strategy.num_replicas_in_sync
        l2_weight_decay = 1e-4
        loss += (tf.reduce_sum(model.losses) / num_replicas)

      minimize_using_explicit_allreduce(
          tape, optimizer, loss, model.trainable_variables)
      #self.train_loss.update_state(loss)
      #self.train_accuracy.update_state(labels, logits)

    strategy.run(step_fn, args=(next(iterator),))

def set_tfconfig(tf_config):
    os.environ["TF_CONFIG"] = json.dumps(tf_config)

def get_handler(minN, maxN):
    tf_config = json.loads(os.environ["TF_CONFIG"])
    server_address = tf_config["cluster"]["orchestrator"][0]
    return orch_api.TFEOrchestratorHandler(server_address, minN, maxN)

@tf.function
def save_internal(checkpoint, wbackup_fdr):
  checkpoint.write(wbackup_fdr)

def save(strategy, backup_fdr, checkpoint, local_address):
  if(strategy.extended._is_chief):
    wbackup_fdr = backup_fdr
  else:
    wbackup_fdr = backup_fdr+local_address.strip().replace(":","")

  save_internal(checkpoint, wbackup_fdr)  

def run_eval(strategy, multi_worker_model, eval_iter, test_loss, test_accuracy, test_dataset, global_batch_size, tf_config):
  test_loss.reset_states()
  test_accuracy.reset_states()

  @tf.function   
  def estep_fn(images, labels):
    """Function to run on the device."""
    logits = multi_worker_model(images, training=True)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
    loss = tf.reduce_sum(loss) * (1.0 / global_batch_size)
    test_loss.update_state(loss)
    test_accuracy.update_state(labels, logits)
    
    

  for image, label in test_dataset:
    start = time.time()
    strategy.run(estep_fn, args=(image, label))
    print(f"Step time took {time.time() - start}")
    #print(multi_worker_model.evaluate(image, label))
  #print(multi_worker_model.evaluate(test_dataset))

  print(f"{tf_config['task']['index']} Loss result is {test_loss.result().numpy()}")
  print(f"{tf_config['task']['index']} Accuracy is {test_accuracy.result().numpy()}")
  

def main(**kwargs):
    """
    The tests should be run assuming we start our process with 
    """
    main_start = time.time()
    
    args = parse_args()
    print(f"Running for {args.epochs} epochs")
    
    strategy = CollectiveAllReduceStrategy()
    
    num_workers = strategy.extended._cluster_resolver.cluster_spec().num_tasks("worker")
    per_worker_batch_size = 64 * 4
    tf_config = json.loads(os.environ["TF_CONFIG"])
    cluster_index = tf_config["task"]["index"]
    print(tf_config)
    tmp_dir = "/tmp/backup"
    s3_path = "s3://sam-debug-binaries/elastic_norestart/old_design"
    if("proc_id" not in kwargs):
      tmp_dir = s3_path
    
    # Here the batch size scales up by number of workers since 
    # `tf.data.Dataset.batch` expects the global batch size. Previously we used 64, 
    # and now this becomes 128.
    global_batch_size = per_worker_batch_size * num_workers
    multi_worker_dataset, test_dataset = cifar_dataset(global_batch_size, args.steps)


    with strategy.scope():
      # Model building/compiling need to be within `strategy.scope()`.
      multi_worker_model, optimizer = build_and_compile_cnn_model()

      #optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

      epoch = tf.Variable(0)

      checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=multi_worker_model, epoch = epoch)


    backup_fdr = tmp_dir+"/chief/test_resnetctl"
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'train_accuracy', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'test_accuracy', dtype=tf.float32)

    @tf.function
    def step_fn(inputs):
      """Function to run on the device."""
      images, labels = inputs
      with tf.GradientTape() as tape:
        logits = multi_worker_model(images, training=True)

        prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits)
        loss = tf.reduce_sum(prediction_loss) * (1.0 / global_batch_size)
        num_replicas = strategy.num_replicas_in_sync
        l2_weight_decay = 1e-4
        loss += (tf.reduce_sum(multi_worker_model.losses) / num_replicas)

      minimize_using_explicit_allreduce(
          tape, optimizer, loss, multi_worker_model.trainable_variables)
      train_loss.update_state(loss)
      train_accuracy.update_state(labels, logits)

    @tf.function
    def train_step(iterator):
      """See base class."""
      strategy.run(step_fn, args=(next(iterator),))

    def ret_eval_dataset(ctx):
      return test_dataset
    eval_dataset = strategy.distribute_datasets_from_function(ret_eval_dataset)

    #Only usable after everything is set in the scope, before running
    ds_data = strategy.experimental_distribute_dataset(multi_worker_dataset)
    epoch_iter = iter(ds_data)

    try: 
      checkpoint.restore(backup_fdr)
    except Exception as e:
      print(e)
    
    print(f"Setup time took {time.time() - main_start}")
    while epoch < args.epochs:
      steps = 0
      epoch_start = time.time()
      while True:
        try:
          start_step = time.time()
          print(f"{cluster_index} | Starting train Step", flush=True)
          train_step(epoch_iter)
          steps +=1
          print(f"{cluster_index} | Current Step is {steps} took {time.time() - start_step} of epoch {epoch.numpy()} and TF_CONFIG is {os.environ['TF_CONFIG']}", flush=True)
        except (tf.errors.OutOfRangeError, StopIteration) as e:
          print(f"End of Epoch {epoch.numpy()} took {time.time() - epoch_start}, restart iterator")
          print(f"Train Loss result is {train_loss.result().numpy()}")
          print(f"Train Accuracy is {train_accuracy.result().numpy()}")
          break
        except Exception as e:
          print(f"Caught exception, total time {time.time() - main_start} lost epoch time {time.time() - epoch_start}", flush=True)
          raise e

      epoch.assign_add(1)
      save(strategy, backup_fdr, checkpoint, str(cluster_index))
      print("Finished Saving", flush=True)
      eval_iter = iter(eval_dataset)
      run_eval(strategy, multi_worker_model, eval_iter, test_loss, test_accuracy, test_dataset, global_batch_size, tf_config)
      epoch_iter = iter(ds_data)
      
    return 0
    

    

if __name__ == "__main__":
    main()

