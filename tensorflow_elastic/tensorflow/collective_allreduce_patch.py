import copy
from six.moves import queue as Queue
import tensorflow as tf
import json, os, time
import weakref
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.framework import c_api_util
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python import pywrap_tfe
from tensorflow.core.framework import function_pb2
from tensorflow.python.util import compat
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.framework import ops
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute.cluster_resolver import ClusterResolver
from tensorflow.core.protobuf import config_pb2
#from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.framework.errors_impl import InternalError as tfInternalError
from tensorflow.python.framework import device as pydev
from tensorflow.python.eager.context import LogicalDevice, ASYNC, _KEEP_ALIVE_SECS

from tensorflow_elastic.tensorflow.tfe_cluster_resolver import TFConfigClusterResolver
from tensorflow.python.framework import errors

from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.core.protobuf import rewriter_config_pb2

def find_iterator(iterator):
  bfs_q = Queue.Queue()
  bfs_q.put(iterator)
  visited = []
  while not bfs_q.empty():
    it = bfs_q.get()
    print(it)
    print("\n")
    print(dir(it), it.__dict__)
    visited.append(it)

    if hasattr(it, "_iterator_resource"):
      print(dir(it._iterator_resource), it._iterator_resource.ref())
    if hasattr(it, "_iterators"):
      for input_iters in it._iterators:
        if input_iters not in visited:
          bfs_q.put(input_iters)
    elif hasattr(it, "_iterator"):
      bfs_q.put(it._iterator)
    
  return it

def nuke_collectives(self, msg):
  context.context().abort_collective_ops(
                  errors.INTERNAL,
                  "Collective Nuke: %s" % msg)

def update_cluster(self, nuke=False):
  #context._reset_context()
  #context.ensure_initialized()
  #tf_config = json.loads(os.environ["TF_CONFIG"])
  #context.context().update_group_size(len(tf_config["cluster"]["worker"]))
  #self._initialize_multi_worker(self._cluster_resolver, True)

  self.nuke_collectives("Updating cluster")
  self._stop_check_health_thread()

  print("*"*20,context.context().get_server_def(), flush=True)
  self._initialize_multi_worker(self._cluster_resolver, True)
  context.context().update_group_size(self._host_cross_device_ops._group_size, self._cross_device_ops._group_size)
  logging.warning("Task %d updated with %d num workers" %
                  (self._cluster_resolver.task_id, self._num_workers))

def __init__(self, container_strategy, cluster_resolver,
               communication_options):
    if not isinstance(communication_options, collective_util.Options):
      raise ValueError("communication_options must be an instance of "
                       "tf.distribute.experimental.CommunicationOptions")
    self._cluster_resolver = cluster_resolver or TFConfigClusterResolver()
    if not isinstance(self._cluster_resolver, ClusterResolver):
      raise ValueError("cluster_resolver must be an instance of "
                       "tf.distribute.cluster_resolver.ClusterResolver")
    distribute_lib.StrategyExtendedV1.__init__(self, container_strategy)
    self._communication_options = communication_options
    self._collective_key_base = container_strategy._collective_key_base  # pylint: disable=protected-access
    self._initialize_strategy(self._cluster_resolver)
    self._cfer_fn_cache = weakref.WeakKeyDictionary()
    self.experimental_enable_get_next_as_optional = True
    assert isinstance(self._cross_device_ops,
                      cross_device_ops_lib.CollectiveAllReduce)

def _initialize_multi_worker(self, cluster_resolver, update_server=False):
    """Initializes the object for multi-worker training."""
    cluster_spec = multi_worker_util.normalize_cluster_spec(
        cluster_resolver.cluster_spec())
    task_type = cluster_resolver.task_type
    task_id = cluster_resolver.task_id
    if task_type is None or task_id is None:
      raise ValueError("When `cluster_spec` is given, you must also specify "
                       "`task_type` and `task_id`.")
    self._cluster_spec = cluster_spec
    self._task_type = task_type
    self._task_id = task_id
    self._id_in_cluster = multi_worker_util.id_in_cluster(
        self._cluster_spec, self._task_type, self._task_id)

    self._num_workers = multi_worker_util.worker_count(cluster_spec, task_type)
    if not self._num_workers:
      raise ValueError("No `worker`, `chief` or `evaluator` tasks can be found "
                       "in `cluster_spec`.")

    self._is_chief = multi_worker_util.is_chief(cluster_spec, task_type,
                                                task_id)

    self._worker_device = "/job:%s/task:%d" % (task_type, task_id)
    self._host_input_device = numpy_dataset.SingleDevice(self._worker_device)

    if (ops.executing_eagerly_outside_functions() and
        not getattr(self, "_local_or_standalone_client_mode", False)):
      context.context().configure_collective_ops(
          collective_leader=multi_worker_util.collective_leader(
              cluster_spec, task_type, task_id),
          scoped_allocator_enabled_ops=("CollectiveReduce",),
          device_filters=("/job:%s/task:%d" % (task_type, task_id),))
      self._collective_ops_configured = True
    
    #If there is a health thread we need to stop it prior to updating the server
    self._stop_check_health_thread()
    # Starting a std server in eager mode and in independent worker mode.
    if (update_server or (context.executing_eagerly() and
        not getattr(self, "_std_server_started", False) and
        not getattr(self, "_local_or_standalone_client_mode", False))):
      # Checking _local_or_standalone_client_mode as well because we should not
      # create the std server in standalone client mode.
      config_proto = copy.deepcopy(context.context().config)
      config_proto = self._update_config_proto(config_proto)

      if hasattr(cluster_resolver, "port"):
        port = cluster_resolver.port
      else:
        port = 0
      print(task_id, cluster_spec, flush=True)
      server_def = tensorflow_server_pb2.ServerDef(
          cluster=cluster_spec.as_cluster_def(),
          default_session_config=config_proto,
          job_name=task_type,
          task_index=task_id,
          protocol=cluster_resolver.rpc_layer or "grpc",
          port=port)
      context.context().enable_collective_ops(server_def)
      self._std_server_started = True
      # The `ensure_initialized` is needed before calling
      # `context.context().devices()`.
      context.context().ensure_initialized()
      logging.info(
          "Enabled multi-worker collective ops with available devices: %r",
          context.context().devices())

    # TODO(yuefengz): The `num_gpus` is only for this particular task. It
    # assumes all workers have the same number of GPUs. We should remove this
    # assumption by querying all tasks for their numbers of GPUs.
    # TODO(b/126786766): TFConfigClusterResolver returns wrong number of GPUs in
    # some cases.
    if isinstance(cluster_resolver, TFConfigClusterResolver):
      num_gpus = context.num_gpus()
    else:
      num_gpus = cluster_resolver.num_accelerators().get("GPU", 0)

    if num_gpus:
      local_devices = tuple("%s/device:GPU:%d" % (self._worker_device, i)
                            for i in range(num_gpus))
    else:
      local_devices = (self._worker_device,)

    self._collective_keys = cross_device_utils.CollectiveKeys(
        group_key_start=1 + self._collective_key_base)
    self._cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(
        devices=local_devices,
        group_size=len(local_devices) * self._num_workers,
        collective_keys=self._collective_keys)
    # CrossDeviceOps for per host tensors.
    self._host_cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(
        devices=[self._worker_device],
        group_size=self._num_workers,
        collective_keys=self._collective_keys)
    super(collective_all_reduce_strategy.CollectiveAllReduceExtended, self)._initialize_single_worker(
        local_devices)

    group_key = self._collective_keys.get_group_key([self._worker_device])
    
    # Add a default device so that ops without specified devices will not end up
    # on other workers.
    self._default_device = "/job:%s/task:%d" % (task_type, task_id)

    # Save the num_gpus_per_worker and rpc_layer for configure method.
    self._num_gpus_per_worker = num_gpus
    self._rpc_layer = cluster_resolver.rpc_layer
    self._warn_nccl_no_gpu()

    #if self._enable_check_health:
      #if(self._num_workers > 1):
    #  self._start_check_health_thread()
        

    logging.info(
        "MultiWorkerMirroredStrategy with cluster_spec = %r, task_type = %r, "
        "task_id = %r, num_workers = %r, local_devices = %r, "
        "communication = %s", cluster_spec.as_dict(), task_type, task_id,
        self._num_workers, local_devices,
        self._communication_options.implementation)


def get_elastic_iterator(self, dataset):
  print("Task %d getting elastic iterator"%(self._task_id))
  it = iter(dataset)
  print(it, type(it), type(dataset))
  self._iterators.append(it)
  
  return it

def _experimental_distribute_dataset(self, dataset, options=None):
  self._input_dataset = dataset
  if (options and options.experimental_replication_mode ==
        distribute_lib.InputReplicationMode.PER_REPLICA):
      raise NotImplementedError(
          "InputReplicationMode.PER_REPLICA "
          "is only supported in "
          "`experimental_distribute_datasets_from_function`."
      )
  input_context = self._make_input_context()
  self._input_dataset_dist = input_lib.get_distributed_dataset(
      dataset,
      self._input_workers_with_options(options),
      self._container_strategy(),
      num_replicas_in_sync=self._num_replicas_in_sync,
      input_context=input_context)
  return self._input_dataset_dist

def _get_variable_creator_initial_value(self,
                                          replica_id,
                                          device,
                                          primary_var,
                                          **kwargs):
    # if(not isinstance(primary_var, type(None))):
    #   print(f"{replica_id}, {device}, {primary_var.name} {primary_var.shape}", flush=True)
    # else:
    #   print(f"{replica_id}, {device}, {kwargs} ", flush=True)

    return super(collective_all_reduce_strategy.CollectiveAllReduceExtended,
                   self)._get_variable_creator_initial_value(
                       replica_id=replica_id,
                       device=device,
                       primary_var=primary_var,
                       **kwargs)

def _make_input_context(self):
    task_ids = self._cluster_spec.task_indices("worker")
    actual_id = task_ids.index(self._task_id)
    #print(f"\n\nTask ids {task_ids} actual id {actual_id}\n\n", flush=True)
    input_context = distribute_lib.InputContext(
        num_input_pipelines=self._num_workers,
        input_pipeline_id=actual_id,
        num_replicas_in_sync=self._num_replicas_in_sync)
    return input_context

def _check_health(self):
    while True:
      if self._check_health_thread_should_stop.is_set():
        return
      for job in self._cluster_spec.jobs:
        for task_id in self._cluster_spec.task_indices("worker"):
          peer = "/job:{}/replica:0/task:{}".format(job, task_id)
          attempts = 0
          while True:
            attempts += 1
            try:
              context.context().check_collective_ops_peer_health(
                  peer, timeout_in_ms=self._check_health_timeout * 1000)
              # If check_collective_ops_peer_health doesn't raise an Exception,
              # the peer is healthy.
              break
            except (errors.UnavailableError, errors.FailedPreconditionError,
                    errors.DeadlineExceededError) as e:
              # TODO(b/151232436): Always raise UnavailableError when a peer
              # fails. Now there could be many kinds of errors:
              # - Unavailable: when the peer is not reachable, e.g. it's down.
              # - FailedPrecondition: when the peer has restarted.
              if attempts < self._check_health_retry_limit:
                logging.warning("%s seems down, retrying %d/%d", peer, attempts,
                                self._check_health_retry_limit)
                continue
              logging.error(
                  "Cluster check alive failed, %s is down, "
                  "aborting collectives: %s", peer, e)
              context.context().abort_collective_ops(
                  errors.UNAVAILABLE,
                  "cluster check alive failed, {} is down".format(peer))
              #raise e
              return
            except Exception as e:  # pylint: disable=broad-except
              logging.error("Unexpected exception in check alive: %s", e)
              context.context().abort_collective_ops(
                  errors.INTERNAL,
                  "unexecpted exception in check alive: %s" % e)
              #raise e
              return
      time.sleep(self._check_health_interval)

from tensorflow.python.ops import array_ops

def read_var(self, replica_local_var):
    """Read the aggregate value of a replica-local variable."""
    # pylint: disable=protected-access
    if distribute_utils.is_sync_on_read(replica_local_var):
      return replica_local_var._get_cross_replica()
    assert distribute_utils.is_mirrored(replica_local_var)
    return array_ops.identity(replica_local_var._get())
    # pylint: enable=protected-access

from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
import traceback

def _reduce_to(self, reduce_op, value, destinations, options):
    #print(reduce_op, value, destinations, flush=True)
    if (isinstance(value, values.Mirrored) and
        reduce_op == reduce_util.ReduceOp.MEAN):
      return value
    assert not isinstance(value, values.Mirrored)

    if (isinstance(value, values.DistributedValues) and
        len(self.worker_devices) == 1):
      value = value.values[0]

    # When there are multiple workers, we need to reduce across workers using
    # collective ops.
    if (not isinstance(value, values.DistributedValues) and
        self._num_workers == 1):
      # This function handles reducing values that are not PerReplica or
      # Mirrored values. For example, the same value could be present on all
      # replicas in which case `value` would be a single value or value could
      # be 0.
      return cross_device_ops_lib.reduce_non_distributed_value(
          reduce_op, value, destinations, len(self.worker_devices))
    return self._get_cross_device_ops(value).reduce(
        reduce_op,
        value,
        destinations=destinations,
        options=self._communication_options.merge(options))

import threading
def _start_check_health_thread(self):
    if not context.executing_eagerly():
      logging.info("Check health is only supported in eager.")
      return
    # Use a dummy all-reduce as a barrier to wait for all workers to be up,
    # otherwise the check health may fail immediately.

    # Use array_ops.identity to create the dummy tensor so that we have a new
    # Tensor. If we use constant it may be a cached from on a /job:localhost
    # device, which will cause some code that relies on tensor.device to error.
    #
    # TODO(b/151232436): change to an explicit barrier if we have it.
    dummy_value = array_ops.identity([])
    #print(f"\n\n\n\n DUMMY VALUE IS {dummy_value}\n\n\n\n", flush=True)
    logging.info("Waiting for the cluster, timeout = %s",
                 self._check_health_initial_timeout or "inf")
    try:
      self._host_cross_device_ops.reduce(
          reduce_util.ReduceOp.SUM,
          dummy_value,
          dummy_value,
          options=collective_util.Options(
              timeout_seconds=self._check_health_initial_timeout,
              implementation=collective_util.CommunicationImplementation.RING))
      if context.is_async():
        context.async_wait()
    except errors.DeadlineExceededError:
      raise RuntimeError(
          "Timeout waiting for the cluster, timeout is %d seconds" %
          self._check_health_initial_timeout)
    logging.info("Cluster is ready.")
    self._check_health_thread_should_stop = threading.Event()
    # Start the thread as daemon to avoid it blocking the program from exiting.
    # We try best to shutdown the thread but __del__ is not guaranteed to be
    # called when program exists.
    self._check_health_thread = threading.Thread(
        target=self._check_health,
        daemon=True)
    self._check_health_thread.start()

#collective_all_reduce_strategy.CollectiveAllReduceExtended._start_check_health_thread = _start_check_health_thread 
collective_all_reduce_strategy.CollectiveAllReduceExtended._reduce_to = _reduce_to 
#collective_all_reduce_strategy.CollectiveAllReduceExtended.read_var = read_var 
collective_all_reduce_strategy.CollectiveAllReduceExtended.nuke_collectives = nuke_collectives 
collective_all_reduce_strategy.CollectiveAllReduceExtended._check_health = _check_health
#collective_all_reduce_strategy.CollectiveAllReduceExtended.get_elastic_iterator = get_elastic_iterator
collective_all_reduce_strategy.CollectiveAllReduceExtended.update_cluster = update_cluster
collective_all_reduce_strategy.CollectiveAllReduceExtended._initialize_multi_worker = _initialize_multi_worker
#collective_all_reduce_strategy.CollectiveAllReduceExtended._experimental_distribute_dataset = _experimental_distribute_dataset
collective_all_reduce_strategy.CollectiveAllReduceExtended.__init__ = __init__
#collective_all_reduce_strategy.CollectiveAllReduceExtended._get_variable_creator_initial_value = _get_variable_creator_initial_value
collective_all_reduce_strategy.CollectiveAllReduceExtended._make_input_context = _make_input_context

from tensorflow.python.distribute import cross_device_ops
from tensorflow.python.distribute import values as value_lib

def reduce(self, reduce_op, per_replica_value, destinations, options=None):
    """Reduce `per_replica_value` to `destinations`.

    See `tf.distribute.StrategyExtended.reduce_to`. This can only be called in
    the cross-replica context.

    Args:
      reduce_op: a `tf.distribute.ReduceOp` specifying how values should be
        combined.
      per_replica_value: a `tf.distribute.DistributedValues`, or a `tf.Tensor`
        like object.
      destinations: a `tf.distribute.DistributedValues`, a `tf.Variable`, a
        `tf.Tensor` alike object, or a device string. It specifies the devices
        to reduce to. To perform an all-reduce, pass the same to `value` and
        `destinations`. Note that if it's a `tf.Variable`, the value is reduced
        to the devices of that variable, and this method doesn't update the
        variable.
      options: a `tf.distribute.experimental.CommunicationOptions`. See
        `tf.distribute.experimental.CommunicationOptions` for details.

    Returns:
      A `tf.Tensor` or `tf.distribute.DistributedValues`.

    Raises:
      ValueError: if per_replica_value can't be converted to a
        `tf.distribute.DistributedValues` or if destinations is not a string,
        `tf.Variable` or `tf.distribute.DistributedValues`.
    """
    if options is None:
      options = collective_util.Options()
    if not isinstance(per_replica_value, value_lib.DistributedValues):
      #print(per_replica_value, flush=True)
      per_replica_value = cross_device_ops._make_tensor_into_per_replica(per_replica_value)
      #print(per_replica_value, flush=True)

    cross_device_ops.validate_destinations(destinations)

    # Shortcut if `per_replica_value` only contains one value.
    # if self._num_between_graph_workers == 1 and len(
    #     per_replica_value.values) == 1 and cross_device_ops._devices_match(
    #         per_replica_value, destinations):
    #   print("Inside Shortcut", flush=True)
    #   with ops.device(per_replica_value.values[0].device):
    #     v = array_ops.identity(per_replica_value.values[0])
    #   return distribute_utils.regroup((v,), wrap_class=value_lib.Mirrored)

    if options is None:
      options = collective_util.Options()
    #print("No Shortcut", flush=True)
    return self.reduce_implementation(reduce_op, per_replica_value,
                                      destinations, options)

cross_device_ops.CrossDeviceOps.reduce = reduce
#from tensorflow.python.distribute.cross_device_utils import CollectiveReplicaLauncher
#CollectiveReplicaLauncher._use_collective_v2 = True


# Update context
def update_group_size(self, host_group_size, device_group_size):
  modified_fns = []
  for each in self._added_fns:
    if (self.has_function(each)):
      f_def = self.get_function_def(each)
      modified = False
      for node in f_def.node_def:
        if (node.op == "CollectiveReduce"):
          modified = True
          #logging.warning(node)
          if("CPU" in node.device):
            group_size = host_group_size
          elif("GPU" in node.device):
            group_size = device_group_size

          for attr_name in node.attr:
            if (attr_name == "group_size"):
              node.attr[attr_name].i = group_size
        if (node.op == "CollectiveBcastRecv" or
            node.op == "CollectiveBcastSend"):
          #logging.warning(node)
          modified = True
          if("CPU" in node.device):
            group_size = host_group_size
          elif("GPU" in node.device):
            group_size = device_group_size
          for attr_name in node.attr:
            if (attr_name == "group_size"):
              node.attr[attr_name].i = group_size

      #print("Function def is ",f_def)
      if (modified):
        self.remove_function(each)
        modified_fns.append((each, f_def))
  for each in modified_fns:
    self._added_fns.remove(each[0])
    self.add_function_def(each[1])

import traceback
def add_function(self, fn):
  """Add a function definition to the context.

  Once added, the function (identified by its name) can be executed like any
  other operation.

  Args:
    fn: A wrapped TF_Function (returned from TF_GraphToFunction_wrapper).
  """
  
  #traceback.print_stack()
  self.ensure_initialized()
  with c_api_util.tf_buffer() as buffer_:
    pywrap_tf_session.TF_FunctionToFunctionDef(fn, buffer_)
    proto_data = pywrap_tf_session.TF_GetBuffer(buffer_)
  function_def = function_pb2.FunctionDef()
  function_def.ParseFromString(compat.as_bytes(proto_data))
  name = compat.as_bytes(function_def.signature.name)
  self._added_fns.append(name)
  pywrap_tfe.TFE_ContextAddFunction(self._handle, fn)


def add_function_def(self, fdef):
  """Add a function definition to the context.

  Once added, the function (identified by its name) can be executed like any
  other operation.

  Args:
    fdef: A FunctionDef protocol buffer message.
  """
  self.ensure_initialized()
  self._added_fns.append(compat.as_bytes(fdef.signature.name))
  fdef_string = fdef.SerializeToString()
  pywrap_tfe.TFE_ContextAddFunctionDef(self._handle, fdef_string,
                                       len(fdef_string))

def ensure_initialized(self):
    """Initialize handle and devices if not already done so."""
    if self._initialized:
      return
    with self._initialize_lock:
      if self._initialized:
        return
      assert self._context_devices is None
      opts = pywrap_tfe.TFE_NewContextOptions()
      try:
        config_str = self.config.SerializeToString()
        pywrap_tfe.TFE_ContextOptionsSetConfig(opts, config_str)
        if self._device_policy is not None:
          pywrap_tfe.TFE_ContextOptionsSetDevicePlacementPolicy(
              opts, self._device_policy)
        if self._mirroring_policy is not None:
          pywrap_tfe.TFE_ContextOptionsSetMirroringPolicy(
              opts, self._mirroring_policy)
        if self._default_is_async == ASYNC:
          pywrap_tfe.TFE_ContextOptionsSetAsync(opts, True)
        if self._lazy_remote_inputs_copy is not None:
          pywrap_tfe.TFE_ContextOptionsSetLazyRemoteInputsCopy(
              opts, self._lazy_remote_inputs_copy)
        if self._use_tfrt is not None:
          pywrap_tfe.TFE_ContextOptionsSetTfrt(opts, self._use_tfrt)
        context_handle = pywrap_tfe.TFE_NewContext(opts)
      finally:
        pywrap_tfe.TFE_DeleteContextOptions(opts)
      assert not (self._server_def and self._collective_ops_server_def), (
          "Cannot enable remote execution as well as collective ops at the "
          "moment. If this is important to you, please file an issue.")
      if self._server_def is not None:
        server_def_str = self._server_def.SerializeToString()
        pywrap_tfe.TFE_ContextSetServerDef(context_handle, _KEEP_ALIVE_SECS,
                                           server_def_str)
      elif self._collective_ops_server_def is not None:
        server_def_str = self._collective_ops_server_def.SerializeToString()
        pywrap_tfe.TFE_EnableCollectiveOps(context_handle, server_def_str)

      self._context_handle = context_handle
      self._initialize_logical_devices()
      self._initialized = True

def _initialize_logical_devices(self):
    """Helper to initialize devices."""
    # Store list of devices
    logical_devices = []
    context_devices = []
    device_list = pywrap_tfe.TFE_ContextListDevices(self._context_handle)
    try:
      self._num_gpus = 0
      for i in range(pywrap_tfe.TF_DeviceListCount(device_list)):
        dev_name = pywrap_tfe.TF_DeviceListName(device_list, i)
        #print(f"Init local devices: ***************{dev_name}*************", flush=True)
        context_devices.append(pydev.canonical_name(dev_name))
        spec = pydev.DeviceSpec.from_string(dev_name)
        # If the job is localhost, we assume that the cluster has not yet been
        # configured and thus clear the job, replica & task.
        if spec.job == "localhost":
          spec = spec.replace(job=None, replica=None, task=None)
        logical_devices.append(
            LogicalDevice(name=spec.to_string(), device_type=spec.device_type))
        dev_type = pywrap_tfe.TF_DeviceListType(device_list, i)
        if dev_type == "GPU":
          self._num_gpus += 1

    finally:
      self._logical_devices = logical_devices
      self._context_devices = context_devices
      pywrap_tfe.TF_DeleteDeviceList(device_list)

def configure_collective_ops(
      self,
      collective_leader="",
      scoped_allocator_enabled_ops=("CollectiveReduce",),
      use_nccl_communication=False,
      device_filters=None):
    """Configure collective ops.

      Collective group leader is necessary for collective ops to run, other
      configurations are mainly for the purpose of performance.

    Args:
      collective_leader: a device string for collective leader, e.g.
        "/job:worker/replica:0/task:0"; empty string means local execution of
          collective ops.
      scoped_allocator_enabled_ops: a tuple or a list of op names for scoped
        allocator to run with.
      use_nccl_communication: whether to use nccl communication for collective
        ops.
      device_filters: a tuple or a list of device strings. If set, corresponding
        task can only see the devices filtered by these device filters.

    Raises:
      RuntimeError: if this method is not called at program startup.
    """
    # if self._collective_leader is not None:
    #   if (self._collective_leader != collective_leader or
    #       self._collective_scoped_allocator_enabled_ops !=
    #       scoped_allocator_enabled_ops or
    #       self._collective_use_nccl_communication != use_nccl_communication or
    #       self._collective_device_filters != device_filters):
    #     raise ValueError("Collective ops are already configured.")
    #   else:
    #     return

    # if self._context_handle is not None:
    #   raise RuntimeError("Collective ops must be configured at program startup")

    self._collective_leader = collective_leader
    self._collective_scoped_allocator_enabled_ops = scoped_allocator_enabled_ops
    self._collective_use_nccl_communication = use_nccl_communication
    self._collective_device_filters = device_filters

def enable_collective_ops(self, server_def):
    """Enable distributed collective ops with an appropriate server_def.
    Args:
      server_def: A tensorflow::ServerDef proto. Enables execution on remote
        devices.
    Raises:
      ValueError: if server_def is None.
      RuntimeError: if this method is not called at program startup.
    """
    if not server_def:
      raise ValueError("server_def is None.")

    self._collective_ops_server_def = server_def
    print("*"*20, server_def, flush=True)

    # TODO(b/129298253): Allow creating datasets/tensors before enabling
    # collective ops.
    if self._context_handle is not None:
      logging.warning("Enabling collective ops after program startup may cause "
                      "error when accessing previously created tensors.")
      with self._initialize_lock:
        assert self._initialized
        server_def_str = self._collective_ops_server_def.SerializeToString()
        pywrap_tfe.TFE_EnableCollectiveOps(self._context_handle, server_def_str)
        self._initialize_logical_devices()
        self._clear_caches()

@property
def config(self):
    """Return the ConfigProto with all runtime deltas applied."""
    # Ensure physical devices have been discovered and config has been imported
    self._initialize_physical_devices()

    config = config_pb2.ConfigProto()
    if self._config is not None:
      config.CopyFrom(self._config)

    if self._optimizer_jit is not None:
      config.graph_options.optimizer_options.global_jit_level = (
          config_pb2.OptimizerOptions.ON_1
          if self._optimizer_jit else config_pb2.OptimizerOptions.OFF)
    if self._intra_op_parallelism_threads is not None:
      config.intra_op_parallelism_threads = self._intra_op_parallelism_threads
    if self._inter_op_parallelism_threads is not None:
      config.inter_op_parallelism_threads = self._inter_op_parallelism_threads

    if self._soft_device_placement is not None:
      config.allow_soft_placement = self._soft_device_placement
    else:
      config.allow_soft_placement = self.executing_eagerly()

    if self._log_device_placement is not None:
      config.log_device_placement = self._log_device_placement

    is_mlir_bridge_enabled = pywrap_tfe.TF_IsMlirBridgeEnabled()
    config.experimental.mlir_bridge_rollout = is_mlir_bridge_enabled
    if (is_mlir_bridge_enabled ==
        config_pb2.ConfigProto.Experimental.MLIR_BRIDGE_ROLLOUT_ENABLED):
      config.experimental.enable_mlir_bridge = True

    if self._enable_mlir_graph_optimization is not None:
      config.experimental.enable_mlir_graph_optimization = (
          self._enable_mlir_graph_optimization)

    def rewriter_toggle(option):
      toggle = self._optimizer_experimental_options.get(option, None)
      if toggle is None:
        return

      setattr(config.graph_options.rewrite_options,
              option,
              (rewriter_config_pb2.RewriterConfig.ON
               if toggle else rewriter_config_pb2.RewriterConfig.OFF))

    def rewriter_bool(option):
      toggle = self._optimizer_experimental_options.get(option, None)
      if toggle is None:
        return

      setattr(config.graph_options.rewrite_options,
              option,
              toggle)

    rewriter_toggle("layout_optimizer")
    rewriter_toggle("constant_folding")
    rewriter_toggle("shape_optimization")
    rewriter_toggle("remapping")
    rewriter_toggle("arithmetic_optimization")
    rewriter_toggle("dependency_optimization")
    rewriter_toggle("loop_optimization")
    rewriter_toggle("function_optimization")
    rewriter_toggle("debug_stripper")
    rewriter_bool("disable_model_pruning")
    rewriter_toggle("scoped_allocator_optimization")
    rewriter_toggle("pin_to_host_optimization")
    rewriter_toggle("implementation_selector")
    rewriter_toggle("auto_mixed_precision")
    rewriter_bool("disable_meta_optimizer")
    nodes = self._optimizer_experimental_options.get("min_graph_nodes", None)
    if nodes is not None:
      config.graph_options.rewrite_options.min_graph_nodes = nodes

    # Compute device counts
    config.device_count["CPU"] = 0
    config.device_count["GPU"] = 0
    for dev in self._physical_devices:
      if dev not in self._visible_device_list:
        continue

      virtual_devices = self._virtual_device_map.get(dev)
      if virtual_devices is None:
        config.device_count[dev.device_type] += 1
      else:
        config.device_count[dev.device_type] += len(virtual_devices)

    # Configure gpu_options
    gpu_options = self._compute_gpu_options()
    config.gpu_options.MergeFrom(gpu_options)

    # Configure collective ops
    if self._collective_leader:
      config.experimental.collective_group_leader = self._collective_leader
    if self._collective_scoped_allocator_enabled_ops:
      rewrite_options = config.graph_options.rewrite_options
      rewrite_options.scoped_allocator_optimization = (
          rewriter_config_pb2.RewriterConfig.ON)
      del rewrite_options.scoped_allocator_opts.enable_op[:]
      for op in self._collective_scoped_allocator_enabled_ops:
        rewrite_options.scoped_allocator_opts.enable_op.append(op)
    if self._collective_use_nccl_communication:
      config.experimental.collective_nccl = True
    if self._collective_device_filters:
      del config.device_filters[:]
      for f in self._collective_device_filters:
        config.device_filters.append(f)

    return config

#context.Context.config = config
context.Context.update_group_size = update_group_size
context.Context.add_function = add_function
context.Context.add_function_def = add_function_def
context.Context._added_fns = []
context.Context.configure_collective_ops = configure_collective_ops
#context.Context._initialize_logical_devices = _initialize_logical_devices
#context.Context.ensure_initialized = ensure_initialized
context.Context.enable_collective_ops = enable_collective_ops


#multi_worker_util.py patches
# TODO(yuefengz): add more validations.
def _validate_cluster_spec(cluster_spec, task_type, task_id):
  """Validates `cluster_spec`.

  It checks:
  0) None of `cluster_spec`, `task_type`, and `task_id` is `None`.
  1) task type is one of "chief", "worker" or "evaluator".
  2) whether there is such a task type as `task_type` in the `cluster_spec`. The
     only exception is `evaluator`. In other words, it is still a valid
     configuration when `task_type` is `evaluator` but it doesn't appear in
     `cluster_spec`. This is to be compatible with `TF_CONFIG` in Estimator.
  3) whether there is at most one "chief" job.
  4) whether there is at most one "evaluator" job.
  5) whether the `task_id` is smaller than the number of tasks for that
     particular `task_type`.

  Args:
    cluster_spec: a dict, `ClusterDef` or `ClusterSpec` object to be validated.
    task_type: string indicating the type of the task.
    task_id: task_id: the id of the `task_type` in this cluster.
  Throws:
    ValueError: if `cluster_spec` fails any check.
  """
  if cluster_spec is None or task_type is None or task_id is None:
    raise ValueError(
        "None of `cluster_spec`, `task_type`, and `task_id` should be `None`.")

  cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_spec).as_dict()
  if task_type not in ("chief", "worker", "evaluator", "ps"):
    raise ValueError(
        "Unrecognized task_type: %r, valid task types are: \"chief\", "
        "\"worker\", \"evaluator\" and \"ps\"." % task_type)

  if task_type and task_type not in cluster_spec and task_type != "evaluator":
    raise ValueError("`task_type` %r not found in cluster_spec." % task_type)

  if len(cluster_spec.get("chief", [])) > 1:
    raise ValueError("There must be at most one 'chief' job.")

  if len(cluster_spec.get("evaluator", [])) > 1:
    raise ValueError("There must be at most one 'evaluator' job.")
  
  if(task_id == -1):
    #Special case, just validate a key
    if (isinstance(cluster_spec[task_type], list)):
      task_id = 0
    elif (isinstance(cluster_spec[task_type], dict)):
      task_id = list(cluster_spec[task_type].keys())[0]
    
  # The `evaluator` job is allowed to be missing in `cluster_spec`.
  if task_type in cluster_spec: 
    if (isinstance(cluster_spec[task_type], list) and task_id >= len(cluster_spec[task_type])):
      raise ValueError(
          "The `task_id` %d exceeds the maximum id of %s." % (task_id, task_type))
    elif (isinstance(cluster_spec[task_type], dict) and task_id not in cluster_spec[task_type]):
      raise ValueError(
          "The `task_id` %d not in cluster: %s" % (task_id, cluster_spec[task_type]))

def worker_count(cluster_spec, task_type):
  """Returns the number of workers in the cluster."""
  _validate_cluster_spec(cluster_spec, task_type, task_id=-1)
  cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_spec).as_dict()

  # Other jobs such as "ps" shouldn't call this function.
  if task_type not in ["chief", "worker", "evaluator"]:
    raise ValueError("Unexpected `task_type` %r" % task_type)

  if task_type == "evaluator":
    # The "evaluator" is in its own cluster or its own partition of a cluster.
    # So we don't have to count "chief" or "worker" if the current task is an
    # "evaluator".
    return len(cluster_spec["evaluator"])
  else:
    # In the non-evaluator case, we return the total number of "chief" and
    # "worker" tasks as the "chief" is also a worker.
    return (len(cluster_spec.get("chief", [])) + len(
        cluster_spec.get("worker", [])))
      
def collective_leader(cluster_spec, task_type, task_id):
  """Return the job name for the leader of for collective ops.

  Args:
    cluster_spec: a dict, `ClusterDef` or `ClusterSpec` object specifying the
      cluster configurations.
    task_type: the task type in the cluster.
    task_id: the task id in the cluster.

  Returns:
    a string indicating the leader job name or empty string if no need to set
    leader job.
  """
  cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_spec)

  # No need to set collective leader for local.
  if not cluster_spec.as_dict():
    return ""

  _validate_cluster_spec(cluster_spec, task_type, task_id)

  # Only one evaluator, so no need to set collective leader.
  if task_type == "evaluator":
    return ""

  # Use chief if chief is in the cluster.
  if "chief" in cluster_spec.jobs:
    return "/job:chief/replica:0/task:0"

  # Use min task index worker if no chief
  assert "worker" in cluster_spec.jobs
  indicies = cluster_spec.task_indices("worker")
  min_indicies = min(indicies)
  ret = "/job:worker/replica:0/task:%d"%(min_indicies)
  print(f"Indicies {indicies}, min is {min_indicies}, returning {ret}")
  
  return ret

def is_chief(cluster_spec=None, task_type=None, task_id=None):
  """Returns whether the given task is chief in the cluster.

  Since there is at most one evaluator and the evaluator itself should be
  independent of the training cluster, the evaluator job is also a chief job on
  its own.

  If this is currently running under a `_WorkerContext` of distribute
  coordinator, the arguments can be omitted as the result is already available.

  Args:
    cluster_spec: a dict, `ClusterDef` or `ClusterSpec` object specifying the
      cluster configurations.
    task_type: the task type in the cluster.
    task_id: the task id in the cluster.

  Returns:
    a boolean indicating whether the given task is chief.

  Raises:
    ValueError: if `task_type` is not in the `cluster_spec` or `task_id` exceeds
      the maximum id of the `task_type`.
  """
  if multi_worker_util.has_worker_context():
    # If a worker context exists, use the value provided by it.
    return dc_context.get_current_worker_context().is_chief

  _validate_cluster_spec(cluster_spec, task_type, task_id)
  raw_cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_spec)
  cluster_spec = raw_cluster_spec.as_dict()

  if task_type == "chief" or task_type == "evaluator":
    return True

  # If chief not in the cluster_spec, use the first worker as chief. This is
  # common in CollectiveAllReduceStrategy.

  # Use min task index worker if no chief
  assert "worker" in raw_cluster_spec.jobs
  indicies = raw_cluster_spec.task_indices("worker")
  min_indicies = min(indicies)
  
  if ("chief" not in cluster_spec and task_type == "worker" and task_id == min_indicies):
    return True
  return False


multi_worker_util.collective_leader = collective_leader
multi_worker_util._validate_cluster_spec = _validate_cluster_spec
multi_worker_util.worker_count = worker_count
multi_worker_util.is_chief = is_chief

from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values as values_lib

def regroup(values, wrap_class=values_lib.PerReplica, always_wrap=False):
  """Makes a nest per-replica into a nest of PerReplica/Mirrored values.

  Args:
    values: Values to regroup
    wrap_class: Class that `values` be wrapped in.
    always_wrap: Always wrap the `values` in `wrap_class` even if the values
        are the same except for DistributeVariable.
  Returns:
    Wrapped `values`.
  """
  v0 = values[0]
  if isinstance(v0, list):
    #for v in values:
    #  print(len(v))
    for v in values[1:]:
      assert isinstance(v, list)
      assert len(v) == len(v0), ("len(v) == %d, len(v0) == %d, v: %s, v0: %s" %
                                 (len(v), len(v0), v, v0))
    return [
        regroup(tuple(v[i] for v in values), wrap_class, always_wrap)
        for i in range(len(v0))
    ]

  if isinstance(v0, tuple):
    for v in values[1:]:
      assert isinstance(v, tuple)
      assert len(v) == len(v0), (len(v), len(v0), wrap_class)
    regrouped_tuple = tuple(
        regroup(tuple(v[i] for v in values), wrap_class, always_wrap)
        for i in range(len(v0)))
    if hasattr(v0, "_fields"):
      # This tuple is in fact a namedtuple! Create a new namedtuple instance
      # and initialize it with the regrouped values:
      assert hasattr(v0, "_make")
      return v0._make(regrouped_tuple)
    else:
      return regrouped_tuple

  if isinstance(v0, dict):
    v0keys = v0.keys()
    for v in values[1:]:
      assert isinstance(v, dict), ("v[0]: %r  v[i]: %r" % (v0, v))
      assert set(v.keys()) == set(v0keys), ("v[0].keys: %s  v[i].keys: %s" %
                                            (set(v0keys), set(v.keys())))
    # Use the actual type in case it is a class inherited from a dict.
    return type(v0)({
        key: regroup(tuple(v[key] for v in values),
                     wrap_class, always_wrap)
        for key in v0keys
    })

  # If exactly the same object across all devices, return it unwrapped.
  same_id = True
  for v in values[1:]:
    if v is not v0:
      same_id = False
      break
  # Consider three cases where same_id is true:
  # * If v0 is a DistributedVariable (a MirroredVariable or
  #   SyncOnReadVariable, and same_id means it is the same across all
  #   devices), we want to return it. We check DistributedVariable
  #   specifically since it can look like it has a
  #   _distributed_container member since its members do.
  if same_id and isinstance(v0, values_lib.DistributedVariable):
    return v0
  # * If v0 is a member of a distributed variable, in which case
  #   hasattr(v0, "_distributed_container") is true, we want to
  #   return the DistributedVariable that contains it using the
  #   _distributed_container logic below. This case can trigger
  #   same_id when there is only one device.
  # * In any other situation, same_id means we return v0 unless `always_wrap` is
  #   true.
  if same_id and not always_wrap and not hasattr(v0, "_distributed_container"):
    return v0

  # Detect the case where each device has a parallel component of the
  # same MirroredVariable (or SyncOnReadVariable). In this case we
  # want to return the containing MirroredVariable, after a bunch of
  # sanity checking. In particular, each component should have the
  # same container, and the devices of the variables should match the
  # keys of the per-replica dictionary.
  if hasattr(v0, "_distributed_container"):
    # pylint: disable=protected-access
    assert not isinstance(v0, values_lib.MirroredVariable), (
        "ids = %s, values = %s" % ([id(v) for v in values], values))
    distributed_container = v0._distributed_container()
    assert distributed_container is not None
    for v in values[1:]:
      assert distributed_container is v._distributed_container()
    return distributed_container
  # pylint: enable=protected-access

  return wrap_class(values)

#distribute_utils.regroup = regroup