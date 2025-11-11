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
#include <stdio.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::UnchangedShape;

REGISTER_OP("KPFusedSparseSegmentReduce")
    .Input("data: float")
    .Input("indices: Tidx")
    .Input("slice_input: int64")
    .Input("begin: int32")
    .Input("begin_1: int32")
    .Attr("combiner: int = 1")  // 0 for SUM, 1 for MEAN
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Output("output: float")
    .Output("slice_output: int32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("KPFusedSparseSegmentReduceNonzero")
    .Input("data: float")
    .Input("indices: Tidx")
    .Input("slice_input: int64")
    .Input("begin: int32")
    .Attr("combiner: int = 1")  // 0 for SUM, 1 for MEAN
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Output("output_shape: int32")
    .Output("output_indices: int32")
    .Output("output_nonzero: float")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("KPFusedEmbeddingPaddingFast")
    .Input("input0: int64")
    .Input("input1: float")
    .Input("input2: int32")
    .Input("input3: int32")
    .Input("pack: int32")
    .Output("output0: int32")
    .Output("output1: int32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle scalar_shape = c->Scalar();
      c->set_output(0, scalar_shape);
      c->set_output(1, scalar_shape);
      return OkStatus();
    });

REGISTER_OP("KPFusedEmbeddingPadding")
    .Input("input0: int64")
    .Input("input1: float")
    .Input("input2: int32")
    .Input("input3: int32")
    .Input("pack: int32")
    .Output("output0: int32")
    .Output("output1: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      ShapeHandle scalar_shape = c->Scalar();
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &out));
      c->set_output(0, scalar_shape);
      c->set_output(1, out);
      return OkStatus();
    });

REGISTER_OP("KPFusedSparseSelect")
    .Input("input_a: int32")
    .Input("input_b: int32")
    .Input("input_c: int32")
    .Input("greater: int32")
    .Input("equal1: int32")
    .Input("equal2: int32")
    .Input("equal3: int32")
    .Output("output_x: float")
    .Output("output_y: float")
    .Output("output_w: float")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("KPFusedSparseReshape")
    .Input("slice_input: int64")
    .Input("begin: int32")
    .Input("new_shape: int64")
    .Input("pack_const: int64")
    .Output("out_indices: int64")
    .Output("out_shape: int64")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("KPFusedSparseDynamicStitch")
    .Input("x: int64")
    .Input("variables: N * float")
    .Output("output: float")
    .Attr("N: int >= 1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("KPFusedGather")
    .Input("data: float")
    .Input("slice_input: int64")
    .Input("begin: int32")
    .Output("out_shape: int64")
    .Output("out_indices: int32")
    .Output("out_data: float")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("KPFusedEmbeddingActionIdGather")
    .Input("input0: Tindices1")
    .Input("input1: float")
    .Input("input2: Tindices2")
    .Input("input3: int32")
    .Input("pack: int32")
    .Attr("Tindices1: {int32, int64} = DT_INT64")
    .Attr("Tindices2: {int32, int64} = DT_INT32")
    .Output("output0: float")
    .SetShapeFn(shape_inference::UnknownShape);
}  // namespace tensorflow