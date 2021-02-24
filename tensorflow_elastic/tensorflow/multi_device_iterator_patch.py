from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.ops import distribute_options
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.data.ops import multi_device_iterator_ops

from tensorflow_elastic.tensorflow.python.tf_extra import tf_extra as tfe_gen_dataset_ops


def _convert_external_state_policy_to_enum(external_state_policy):
  if external_state_policy == "warn":
    return distribute_options.ExternalStatePolicy.WARN
  if external_state_policy == "ignore":
    return distribute_options.ExternalStatePolicy.IGNORE
  if external_state_policy == "fail":
    return distribute_options.ExternalStatePolicy.FAIL
  raise ValueError(
      "Failed to convert {} to an instance of ExternalStatePolicy."
      "Supported values include: 'warn', 'ignore' and 'fail'".format(
          external_state_policy))


def make_saveable_from_iterator(iterator, external_state_policy="fail"):

  policy_enum = _convert_external_state_policy_to_enum(external_state_policy)
  return _MultiDeviceIteratorSavable(  # pylint: disable=protected-access
      iterator._iterator_resource,  # pylint: disable=protected-access
      iterator._iterator_resource.name,  # pylint: disable=protected-access
      external_state_policy=policy_enum)


class _PerDeviceGenerator(dataset_ops.DatasetV2):
  """A `dummy` generator dataset."""

  def __init__(self, shard_num, multi_device_iterator_resource, incarnation_id,
               source_device, element_spec):
    self._element_spec = element_spec

    multi_device_iterator_string_handle = (
        tfe_gen_dataset_ops.e_multi_device_iterator_to_string_handle(
            multi_device_iterator_resource))

    # TODO(b/124254153): Enable autograph once the overhead is low enough.
    @function.defun(autograph=False)  # Pure graph code.
    def _init_func():
      return multi_device_iterator_string_handle

    init_func_concrete = _init_func.get_concrete_function()

    # TODO(b/124254153): Enable autograph once the overhead is low enough.
    @function.defun(autograph=False)  # Pure graph code.
    def _remote_init_func():
      return functional_ops.remote_call(target=source_device,
                                        args=init_func_concrete.captured_inputs,
                                        Tout=[dtypes.string],
                                        f=init_func_concrete)

    self._init_func = _remote_init_func.get_concrete_function()
    self._init_captured_args = self._init_func.captured_inputs

    # TODO(b/124254153): Enable autograph once the overhead is low enough.
    @function.defun(input_signature=[tensor_spec.TensorSpec([], dtypes.string)],
                    autograph=False)  # Pure graph code.
    def _next_func(string_handle):
      # pylint: disable=protected-access
      multi_device_iterator = (
          tfe_gen_dataset_ops.e_multi_device_iterator_from_string_handle(
              string_handle=string_handle,
              output_types=structure.get_flat_tensor_types(self._element_spec),
              output_shapes=structure.get_flat_tensor_shapes(
                  self._element_spec)))
      return tfe_gen_dataset_ops.e_multi_device_iterator_get_next_from_shard(
          multi_device_iterator=multi_device_iterator,
          shard_num=shard_num,
          incarnation_id=incarnation_id,
          output_types=structure.get_flat_tensor_types(self._element_spec),
          output_shapes=structure.get_flat_tensor_shapes(self._element_spec))

    next_func_concrete = _next_func.get_concrete_function()

    # TODO(b/124254153): Enable autograph once the overhead is low enough.
    @function.defun_with_attributes(
        input_signature=[tensor_spec.TensorSpec([], dtypes.string)],
        attributes={"experimental_ints_on_device": True},
        autograph=False)  # Pure graph code.
    def _remote_next_func(string_handle):
      return functional_ops.remote_call(
          target=source_device,
          args=[string_handle] + next_func_concrete.captured_inputs,
          Tout=structure.get_flat_tensor_types(self._element_spec),
          f=next_func_concrete)

    self._next_func = _remote_next_func.get_concrete_function()
    self._next_captured_args = self._next_func.captured_inputs

    self._incarnation_id_index = -1
    for i, arg in enumerate(self._next_captured_args):
      if arg is incarnation_id:
        self._incarnation_id_index = i

    # TODO(b/124254153): Enable autograph once the overhead is low enough.
    @function.defun(input_signature=[tensor_spec.TensorSpec([], dtypes.string)],
                    autograph=False)  # Pure graph code.
    def _finalize_func(unused_string_handle):
      return array_ops.constant(0, dtypes.int64)

    finalize_func_concrete = _finalize_func.get_concrete_function()

    # TODO(b/124254153): Enable autograph once the overhead is low enough.
    @function.defun(input_signature=[tensor_spec.TensorSpec([], dtypes.string)],
                    autograph=False)  # Pure graph code.
    def _remote_finalize_func(string_handle):
      return functional_ops.remote_call(target=source_device,
                                        args=[string_handle] +
                                        finalize_func_concrete.captured_inputs,
                                        Tout=[dtypes.int64],
                                        f=finalize_func_concrete)

    self._finalize_func = _remote_finalize_func.get_concrete_function()
    self._finalize_captured_args = self._finalize_func.captured_inputs

    variant_tensor = gen_dataset_ops.generator_dataset(
        self._init_captured_args,
        self._next_captured_args,
        self._finalize_captured_args,
        init_func=self._init_func,
        next_func=self._next_func,
        finalize_func=self._finalize_func,
        **self._flat_structure)
    super(_PerDeviceGenerator, self).__init__(variant_tensor)

  def _inputs(self):
    # TODO(b/116506223): Determine which datasets should be used as inputs here.
    return []

  @property
  def element_spec(self):
    return self._element_spec


class _ReincarnatedPerDeviceGenerator(dataset_ops.DatasetV2):
  """Creates a _PerDeviceGenerator-like dataset with a new incarnation_id.

  Re-uses the functions from the provided per_device_dataset and just switches
  out the function argument corresponding to the incarnation_id.
  """

  def __init__(self, per_device_dataset, incarnation_id):
    # pylint: disable=protected-access
    self._element_spec = per_device_dataset.element_spec
    self._init_func = per_device_dataset._init_func
    self._init_captured_args = self._init_func.captured_inputs

    self._next_func = per_device_dataset._next_func
    self._next_captured_args = per_device_dataset._next_captured_args
    # The captured arguments to the next_func are string_handle, incarnation_id.
    # We update the incarnation id to the new one.
    self._next_captured_args[
        per_device_dataset._incarnation_id_index] = incarnation_id

    self._finalize_func = per_device_dataset._finalize_func
    self._finalize_captured_args = per_device_dataset._finalize_captured_args

    variant_tensor = gen_dataset_ops.generator_dataset(
        self._init_captured_args,
        self._next_captured_args,
        self._finalize_captured_args,
        init_func=self._init_func,
        next_func=self._next_func,
        finalize_func=self._finalize_func,
        **self._flat_structure)
    super(_ReincarnatedPerDeviceGenerator, self).__init__(variant_tensor)

  def _inputs(self):
    # TODO(b/116506223): Determine which datasets should be used as inputs here.
    return []

  @property
  def element_spec(self):
    return self._element_spec


def _create_device_dataset(prototype_ds, incarnation_id, prefetch_buffer_size,
                           experimental_slack):
  """Uses _prototype_device_datasets[i] to build a dataset for the device."""
  ds = _ReincarnatedPerDeviceGenerator(prototype_ds, incarnation_id)
  if prefetch_buffer_size > 0:
    if experimental_slack:
      ds = dataset_ops.PrefetchDataset(ds, prefetch_buffer_size, slack_period=1)
    else:
      ds = ds.prefetch(prefetch_buffer_size)
  # TODO(jsimsa): Enable auto-tuning and optimizations when supported for
  # non-CPU devices.
  options = dataset_ops.Options()
  options.experimental_optimization.apply_default_optimizations = False
  options.experimental_optimization.autotune = False
  ds = ds.with_options(options)
  return ds


class MultiDeviceIterator(object):
  """An iterator over multiple devices."""

  def __init__(self,
               dataset,
               devices,
               max_buffer_size=1,
               prefetch_buffer_size=1,
               source_device="/cpu:0"):
    """Constructs a MultiDeviceIterator.

    Args:
      dataset: The input dataset to be iterated over.
      devices: The list of devices to fetch data to.
      max_buffer_size: Maximum size of the host side per device buffer to keep.
      prefetch_buffer_size: if > 0, then we setup a buffer on each device to
        prefetch into.
      source_device: The host device to place the `dataset` on.  In order to
        prevent deadlocks, if the prefetch_buffer_size is greater than the
        max_buffer_size, we set the max_buffer_size to prefetch_buffer_size.
    """
    print("CUSTOM MULTI DEVICE ITERATOR", flush=True)
    options = dataset_ops.Options()
    options.experimental_distribute.num_devices = len(devices)
    dataset = dataset.with_options(options)
    self._dataset = dataset._apply_options()  # pylint: disable=protected-access
    self._experimental_slack = dataset.options().experimental_slack
    self._devices = devices
    self._source_device = source_device
    self._source_device_tensor = ops.convert_to_tensor(source_device)
    self._max_buffer_size = max_buffer_size
    self._prefetch_buffer_size = prefetch_buffer_size

    if self._prefetch_buffer_size > self._max_buffer_size:
      self._max_buffer_size = self._prefetch_buffer_size

    # Create the MultiDeviceIterator.
    with ops.device(self._source_device):
      # TODO(b/121378567): Get rid of this shared_name hack.
      shared_name = ""
      if context.executing_eagerly():
        shared_name = context.shared_name()
      self._multi_device_iterator_resource = (
          tfe_gen_dataset_ops.e_multi_device_iterator(
              devices=self._devices,
              shared_name=shared_name,
              container="",
              **self._dataset._flat_structure))  # pylint: disable=protected-access
      if context.executing_eagerly():
        # Delete the resource when this object is deleted
        self._resource_deleter = resource_variable_ops.EagerResourceDeleter(
            handle=self._multi_device_iterator_resource,
            handle_device=self._source_device)

      # The incarnation ID is used to ensure consistency between the per-device
      # iterators and the multi-device iterator.
      self._incarnation_id = tfe_gen_dataset_ops.e_multi_device_iterator_init(
          self._dataset._variant_tensor,  # pylint: disable=protected-access
          self._multi_device_iterator_resource,
          max_buffer_size=self._max_buffer_size)

    self._prototype_device_datasets = []
    for i, device in enumerate(self._devices):
      with ops.device(device):
        ds = _PerDeviceGenerator(i, self._multi_device_iterator_resource,
                                 self._incarnation_id,
                                 self._source_device_tensor,
                                 self._dataset.element_spec)
        self._prototype_device_datasets.append(ds)

    # TODO(rohanj): Explore the possibility of the MultiDeviceIterator to
    # initialize the device side of the pipeline. This would allow the
    # MultiDeviceIterator to choose, for example, to move some transformations
    # into the device side from its input. It might be useful in rewriting.
    # Create the per device iterators.
    self._device_iterators = []
    for i, device in enumerate(self._devices):
      with ops.device(device):
        ds = _create_device_dataset(self._prototype_device_datasets[i],
                                    self._incarnation_id,
                                    self._prefetch_buffer_size,
                                    self._experimental_slack)
        if context.executing_eagerly():
          self._device_iterators.append(dataset_ops.make_one_shot_iterator(ds))
        else:
          self._device_iterators.append(
              dataset_ops.make_initializable_iterator(ds))

    if not context.executing_eagerly():
      device_iterator_initializers = [
          iterator.initializer for iterator in self._device_iterators
      ]
      self._initializer = control_flow_ops.group(*device_iterator_initializers)

  def _create_device_dataset(self, i):
    """Uses _prototype_device_datasets[i] to build a dataset for the device."""
    ds = self._prototype_device_datasets[i]
    ds = _ReincarnatedPerDeviceGenerator(ds, self._incarnation_id)
    if self._prefetch_buffer_size > 0:
      if self._experimental_slack:
        ds = dataset_ops.PrefetchDataset(ds,
                                         self._prefetch_buffer_size,
                                         slack_period=1)
      else:
        ds = ds.prefetch(self._prefetch_buffer_size)
    # TODO(jsimsa): Enable auto-tuning and optimizations when supported for
    # non-CPU devices.
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.autotune = False
    ds = ds.with_options(options)
    return ds

  def get_next(self, device=None):
    """Returns the next element given a `device`, else returns all in a list."""
    if device is not None:
      index = self._devices.index(device)
      return self._device_iterators[index].get_next()

    result = []
    for i, device in enumerate(self._devices):
      with ops.device(device):
        result.append(self._device_iterators[i].get_next())
    return result

  def get_next_as_optional(self):
    result = []
    for i, device in enumerate(self._devices):
      with ops.device(device):
        result.append(self._device_iterators[i].get_next_as_optional())
    return result

  @property
  def initializer(self):
    if context.executing_eagerly():
      return control_flow_ops.no_op()
    return self._initializer

  def _eager_reset(self):
    """Resets the MultiDeviceIterator in eager mode."""
    if not ops.executing_eagerly_outside_functions():
      raise ValueError("Eager reset is only supported in eager mode.")
    # pylint: disable=protected-access
    self._incarnation_id = tfe_gen_dataset_ops.e_multi_device_iterator_init(
        self._dataset._variant_tensor,
        self._multi_device_iterator_resource,
        max_buffer_size=self._max_buffer_size)
    for i, device in enumerate(self._devices):
      with ops.device(device):
        ds = _create_device_dataset(self._prototype_device_datasets[i],
                                    self._incarnation_id,
                                    self._prefetch_buffer_size,
                                    self._experimental_slack)
        # Reset the device iterator resources with the new dataset.
        ds_variant = ds._variant_tensor
        gen_dataset_ops.make_iterator(
            ds_variant, self._device_iterators[i]._iterator_resource)

  @property
  def element_spec(self):
    return self._dataset.element_spec

  @property
  def _iterator_resource(self):
    return self._multi_device_iterator_resource


class _MultiDeviceIteratorSavable(BaseSaverBuilder.SaveableObject):
  """SaveableObject for saving/restoring multi device iterator state."""

  def __init__(
      self,
      iterator_resource,
      name,
      external_state_policy=distribute_options.ExternalStatePolicy.FAIL):
    serialized_iterator = tfe_gen_dataset_ops.e_serialize_multi_device_iterator(
        iterator_resource, external_state_policy=external_state_policy.value)
    specs = [
        BaseSaverBuilder.SaveSpec(serialized_iterator,
                                  "",
                                  name + "_STATE",
                                  device=iterator_resource.device)
    ]
    super(_MultiDeviceIteratorSavable, self).__init__(iterator_resource, specs,
                                                      name)

  def restore(self, restored_tensors, restored_shapes):
    with ops.colocate_with(self.op):
      return tfe_gen_dataset_ops.e_deserialize_multi_device_iterator(
          self.op, restored_tensors[0])


multi_device_iterator_ops.MultiDeviceIterator = MultiDeviceIterator
multi_device_iterator_ops.make_saveable_from_iterator = make_saveable_from_iterator