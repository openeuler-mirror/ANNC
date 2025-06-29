#include <iostream>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/fingerprint.h"
using namespace tensorflow;

REGISTER_OP("KPFusedLinearEmbeddingWithHashBucket")
    .Input("placeholder: string")
    .Input("resource: resource")
    .Attr("num_buckets: int >= 1")
    .Output("output: float")
    .SetShapeFn(shape_inference::UnknownShape);

class KPFusedLinearEmbeddingWithHashBucketOp : public OpKernel {
 public:
  explicit KPFusedLinearEmbeddingWithHashBucketOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_buckets", &num_buckets_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    // Get tensor from resource
    Var* var = nullptr;
    OP_REQUIRES_OK(context,
                   LookupResource(context, HandleFromInput(context, 1), &var));
    const Tensor& variable_tensor = *var->tensor();

    auto input = input_tensor.flat<tstring>();
    auto variable = variable_tensor.matrix<float>();
    int64 dim_0 = input.size();
    int64 dim_1 = variable.dimension(1);

    std::vector<float> output(dim_0 * dim_1, 0);

    for (int64 i = 0; i < input.size(); ++i) {
      if (input(i) != "") {
        uint64 hash_value = Fingerprint64(input(i).data());
        int64 bucket_idx = hash_value % num_buckets_;
        if (bucket_idx >= 0 && bucket_idx < variable.dimension(0)) {
          int64 base = i * dim_1;
          for (int64 j = 0; j < dim_1; ++j) {
            output[base + j] = variable(bucket_idx, j);
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

    auto output_flat = output_tensor->flat<float>();
    std::copy(output.begin(), output.end(), output_flat.data());
  }

 private:
  int64_t num_buckets_;
};

REGISTER_KERNEL_BUILDER(
    Name("KPFusedLinearEmbeddingWithHashBucket").Device(DEVICE_CPU),
    KPFusedLinearEmbeddingWithHashBucketOp);