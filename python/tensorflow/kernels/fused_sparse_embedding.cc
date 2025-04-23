#include <iostream>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/fingerprint.h"
using namespace tensorflow;

REGISTER_OP("KPFusedSparseEmbedding")
    .Input("placeholder: string")
    .Attr("num_buckets: int >= 1")
    .Output("output_indices: int64")
    .Output("output_values: int64")
    .Output("empty_row_indicator: bool")
    .Output("output_shape: int64")
    .SetShapeFn(shape_inference::UnknownShape);

class KPFusedSparseEmbeddingOp : public OpKernel {
 public:
  explicit KPFusedSparseEmbeddingOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_buckets", &num_buckets_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<tstring>();

    std::vector<std::vector<int64_t>> indices;
    std::vector<int64_t> value;
    std::vector<bool> empty_value;

    for (int64_t i = 0; i < input.size(); ++i) {
      if (input(i) != "") {
        indices.push_back({i, 0});
        uint64_t hash_value = Fingerprint64(input(i).data());
        int64_t x = hash_value % num_buckets_;
        if (x > 0) {
          value.push_back(x);
          empty_value.push_back(false);
        }
      } else {
        indices.push_back({i, 0});
        value.push_back(0);
        empty_value.push_back(true);
      }
    }

    Tensor* output_0 = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({static_cast<int>(indices.size()), 2}), &output_0));
    auto output_matrix_0 = output_0->matrix<int64>();

    Tensor* output_1 = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            1, TensorShape({static_cast<int>(value.size())}), &output_1));

    Tensor* output_2 = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            2, TensorShape({static_cast<int>(empty_value.size())}), &output_2));

    Tensor* output_3 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(3, TensorShape({2}), &output_3));
    output_3->flat<int64_t>()(0) = input.size();
    output_3->flat<int64_t>()(1) = 1;
    for (size_t i = 0; i < indices.size(); ++i) {
      output_matrix_0(i, 0) = indices[i][0];
      output_matrix_0(i, 1) = indices[i][1];
      output_1->flat<int64_t>()(i) = value[i];
      output_2->flat<bool>()(i) = empty_value[i];
    }
  }

 private:
  int64_t num_buckets_;
};

REGISTER_KERNEL_BUILDER(Name("KPFusedSparseEmbedding").Device(DEVICE_CPU),
                        KPFusedSparseEmbeddingOp);