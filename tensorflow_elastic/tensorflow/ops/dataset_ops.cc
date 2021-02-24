/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow
{

  // --------------------------------------------------------------------------

  // The ops in this section can be composed to define an input
  // pipeline. Each op produces a DT_VARIANT tensor that represents
  // a DAG of "dataset" objects. An "dataset" object can be converted
  // to a stateful "iterator" by passing the "dataset" to the
  // "MakeIterator" op.
  //
  // TODO(b/123753214): DT_VARIANT tensors that represent "dataset" objects are
  // not presently serializable. To avoid issues with constant folding, ensure
  // that any "source dataset" ops (i.e. ops that output a dataset and do not
  // take one as input) are marked "stateful".

  REGISTER_OP("EDeleteMultiDeviceIterator")
      .Input("multi_device_iterator: resource")
      .Input("iterators: N * resource")
      .Input("deleter: variant")
      .Attr("N: int >= 0")
      .SetShapeFn(shape_inference::NoOutputs);

  REGISTER_OP("EAnonymousMultiDeviceIterator")
      .Output("handle: resource")
      .Output("deleter: variant")
      .Attr("devices: list(string) >= 1")
      .Attr("output_types: list(type) >= 1")
      .Attr("output_shapes: list(shape) >= 1")
      .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->Scalar());
        c->set_output(1, c->Scalar());
        return Status::OK();
      });

  REGISTER_OP("EMultiDeviceIterator")
      .Output("handle: resource")
      .Attr("devices: list(string) >= 1")
      .Attr("shared_name: string")
      .Attr("container: string")
      .Attr("output_types: list(type) >= 1")
      .Attr("output_shapes: list(shape) >= 1")
      .SetShapeFn(shape_inference::ScalarShape);

  REGISTER_OP("EMultiDeviceIteratorInit")
      .Input("dataset: variant")
      .Input("multi_device_iterator: resource")
      .Input("max_buffer_size: int64")
      .Output("incarnation_id: int64")
      .SetShapeFn(shape_inference::ScalarShape);

  REGISTER_OP("EMultiDeviceIteratorGetNextFromShard")
      .Input("multi_device_iterator: resource")
      .Input("shard_num: int32")
      .Input("incarnation_id: int64")
      .Output("components: output_types")
      .Attr("output_types: list(type) >= 1")
      .Attr("output_shapes: list(shape) >= 1")
      .SetShapeFn(shape_inference::DatasetIteratorShape);

  REGISTER_OP("EMultiDeviceIteratorToStringHandle")
      .Input("multi_device_iterator: resource")
      .Output("string_handle: string")
      .SetShapeFn(shape_inference::ScalarShape);

  REGISTER_OP("EMultiDeviceIteratorFromStringHandle")
      .Input("string_handle: string")
      .Output("multi_device_iterator: resource")
      .Attr("output_types: list(type) >= 0 = []")
      .Attr("output_shapes: list(shape) >= 0 = []")
      .SetShapeFn(shape_inference::ScalarShape);

  REGISTER_OP("ESerializeMultiDeviceIterator")
      .Input("resource_handle: resource")
      .Attr("external_state_policy: int = 0")
      .Output("serialized: variant")
      .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->Vector(c->UnknownDim()));
        return Status::OK();
      });

  REGISTER_OP("EDeserializeMultiDeviceIterator")
      .Input("resource_handle: resource")
      .Input("serialized: variant")
      .SetShapeFn(shape_inference::NoOutputs);

} // namespace tensorflow
