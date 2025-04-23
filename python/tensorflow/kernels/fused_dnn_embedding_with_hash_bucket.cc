#include <iostream>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/fingerprint.h"
using namespace tensorflow;

REGISTER_OP("KPFusedDnnEmbeddingWithHashBucket")
    .Input("placeholder: string")
    .Input("variable: float")
    .Attr("num_buckets: int >= 1")
    .Output("output: float")
    .SetShapeFn(shape_inference::UnknownShape);

class KPFusedDnnEmbeddingWithHashBucketOp : public OpKernel {
 public:
  explicit KPFusedDnnEmbeddingWithHashBucketOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_buckets", &num_buckets_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& variable_tensor = context->input(1);
    auto input = input_tensor.flat<tstring>();
    auto variable = variable_tensor.matrix<float>();
    int64 dim_0 = input.size();
    int64 dim_1 = variable.dimension(1);

    std::vector<std::vector<float>> output(dim_0, std::vector<float>(dim_1, 0));

    for (int64 i = 0; i < input.size(); ++i) {
      if (input(i) != "") {
        uint64 hash_value = Fingerprint64(input(i).data());
        int64 x = hash_value % num_buckets_;
        if (x >= 0 && x < variable.dimension(0)) {
          for (int64 j = 0; j < dim_1; ++j) {
            output[i][j] = variable(x, j);
          }
        }
      } else {
        // Already initialized to zero, no action needed
      }
    }

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({dim_0, dim_1}),
                                            &output_tensor));

    auto output_matrix = output_tensor->matrix<float>();
    for (int64 i = 0; i < dim_0; ++i) {
      for (int64 j = 0; j < dim_1; ++j) {
        output_matrix(i, j) = output[i][j];
      }
    }
  }

 private:
  int64_t num_buckets_;
};

REGISTER_KERNEL_BUILDER(Name("KPFusedDnnEmbeddingWithHashBucket").Device(DEVICE_CPU),
                        KPFusedDnnEmbeddingWithHashBucketOp);