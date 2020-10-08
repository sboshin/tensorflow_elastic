import copy
import tensorflow as tf
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
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import input_lib
from tensorflow.core.protobuf import tensorflow_server_pb2


def update_cluster(self):
  #context._reset_context()
  #context.ensure_initialized()
  self._initialize_multi_worker(self._cluster_resolver, True)
  context.context().update_group_size(self._num_workers)
  logging.warning("Task %d updated with %d num workers"%(self._cluster_resolver.task_id, self._num_workers))

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

  if(not update_server):
  #if(True):
    if (ops.executing_eagerly_outside_functions() and
        not getattr(self, "_local_or_standalone_client_mode", False)):
      context.context().configure_collective_ops(
          collective_leader=multi_worker_util.collective_leader(
              cluster_spec, task_type, task_id),
          scoped_allocator_enabled_ops=("CollectiveReduce",),
          device_filters=("/job:%s/task:%d" % (task_type, task_id),))
      self._collective_ops_configured = True
  else:
    context.context()._collective_device_filters = ("/job:%s/task:%d" % (task_type, task_id),)

  print(context.context()._collective_device_filters, flush=True)
  # Starting a std server in eager mode and in independent worker mode.
  if (update_server or context.executing_eagerly() and
      not getattr(self, "_std_server_started", False) and
      not getattr(self, "_local_or_standalone_client_mode", False)):
    # Checking _local_or_standalone_client_mode as well because we should not
    # create the std server in standalone client mode.
    config_proto = copy.deepcopy(context.context().config)
    config_proto = self._update_config_proto(config_proto)

    if hasattr(cluster_resolver, "port"):
      port = cluster_resolver.port
    else:
      port = 0
    server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_spec.as_cluster_def(),
        default_session_config=config_proto,
        job_name=task_type,
        task_index=task_id,
        protocol=cluster_resolver.rpc_layer or "grpc",
        port=port)
    context.context().enable_collective_ops(server_def)
    print(f"\n\n\nServer Def is {server_def}\n\n",flush=True)
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

  print(f"\n\n\n{context.context().list_logical_devices()}\n\n",flush=True)
  if(not update_server):
    self._collective_keys = cross_device_utils.CollectiveKeys()
  self._cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(
      devices=local_devices,
      group_size=len(local_devices) * self._num_workers,
      collective_keys=self._collective_keys,
      communication=self._communication)
  # CrossDeviceOps for per host tensors.
  self._host_cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(
      devices=[self._worker_device],
      group_size=self._num_workers,
      collective_keys=self._collective_keys,
      communication=cross_device_ops_lib.CollectiveCommunication.RING,
  )
  super(collective_all_reduce_strategy.CollectiveAllReduceExtended, self)._initialize_single_worker(
      local_devices)

  # Add a default device so that ops without specified devices will not end up
  # on other workers.
  self._default_device = "/job:%s/task:%d" % (task_type, task_id)

  # Save the num_gpus_per_worker and rpc_layer for configure method.
  self._num_gpus_per_worker = num_gpus
  self._rpc_layer = cluster_resolver.rpc_layer
  self._warn_nccl_no_gpu()

  # TODO(b/151232436): Enable check health thread by default.
  if self._enable_check_health:
    self._start_check_health_thread()

  logging.info(
      "MultiWorkerMirroredStrategy with cluster_spec = %r, task_type = %r, "
      "task_id = %r, num_workers = %r, local_devices = %r, "
      "communication = %s", cluster_spec.as_dict(), task_type,
      task_id, self._num_workers, local_devices,
      self._communication)



collective_all_reduce_strategy.CollectiveAllReduceExtended.update_cluster = update_cluster
collective_all_reduce_strategy.CollectiveAllReduceExtended._initialize_multi_worker = _initialize_multi_worker

# Update context
def update_group_size(self, group_size):
    modified_fns = []
    for each in self._added_fns:
      if(self.has_function(each)):
        f_def = self.get_function_def(each)
        modified = False
        for node in f_def.node_def:
          if(node.op == "CollectiveReduce"):
            modified = True
            for attr_name in node.attr:
              if(attr_name == "group_size"):
                node.attr[attr_name].i = group_size
          if(node.op == "CollectiveBcastRecv" or node.op == "CollectiveBcastSend"):
            # logging.warning(node)
            modified = True
            for attr_name in node.attr:
              if(attr_name == "group_size"):
                node.attr[attr_name].i = group_size
          
        #print("Function def is ",f_def)
        if(modified):
          self.remove_function(each)
          modified_fns.append((each, f_def))
    for each in modified_fns:
      self._added_fns.remove(each[0])
      self.add_function_def(each[1])

def add_function(self, fn):
  """Add a function definition to the context.

  Once added, the function (identified by its name) can be executed like any
  other operation.

  Args:
    fn: A wrapped TF_Function (returned from TF_GraphToFunction_wrapper).
  """
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

context.Context.update_group_size = update_group_size
context.Context.add_function = add_function
context.Context.add_function_def = add_function_def
context.Context._added_fns = []
