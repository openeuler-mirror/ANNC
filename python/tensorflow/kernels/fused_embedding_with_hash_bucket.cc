#include <iostream>
#include <vector>

#include "farmhash.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"

using namespace tensorflow;

REGISTER_OP("KPFusedEmbeddingWithHashBucket")
    .Input("palcehodler: string")
    .Input("variable: T_weight")
    .Attr("num_buckets: int >= 1")
    .Attr("combiner: int")
    .Attr("T_weight: {resource, float}")
    .Output("output: float")
    .SetShapeFn(shape_inference::UnknownShape);

class KPFusedEmbeddingWithHashBucketOp : public OpKernel {
    public:
      explicit KPFusedEmbeddingWithHashBucketOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("num_buckets", &num_buckets_));}

      void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        float *weight;
        float *output;
        const Tensor& input_tensor = context->input(0);
        const Tensor* weight_tensor = &context->input(1);
        
        // for saved model
        if (weight_tensor->dtype() == DT_RESOURCE) {
          Var* variable;
          OP_REQUIRES_OK(context,
                        LookupResource(context, HandleFromInput(context, 1), 
                                        &variable));
          core::ScopedUnref s(variable);
          weight_tensor = variable->tensor();
          OP_REQUIRES(context, weight_tensor->dtype() == DT_FLOAT,
                      errors::InvalidArgument("Expect float weight"));
        }
        // auto start = std::chrono::high_resolution_clock::now();
        
        auto input = input_tensor.flat<tstring>();
        weight = (float *)weight_tensor->tensor_data().data();
        int64_t batch = input_tensor.dim_size(0);
        int64_t embedding_size = weight_tensor->dim_size(1);
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, 
                       context->allocate_output(
                        0, TensorShape({batch, embedding_size}), 
                        &output_tensor));
        output = (float *)output_tensor->tensor_data().data();
        memset(output, 0, batch * embedding_size * sizeof(float));
        if (embedding_size == 1) {
          for (int64 i = 0; i < batch; ++i) {
            if(TF_PREDICT_TRUE(!input(i).empty())) {
              uint64 hash_value = util::Fingerprint64(input(i).data(), input(i).length());
              uint64 x = hash_value % num_buckets_;
              output[i] = weight[x];
            }
          }
        } else {
          for (int64 i = 0; i < batch; ++i) {
            if(TF_PREDICT_TRUE(!input(i).empty())) {
              uint64 hash_value = util::Fingerprint64(input(i).data(), input(i).length());
              uint64 x = hash_value % num_buckets_;
              memcpy(output + i * embedding_size, weight + x * embedding_size, embedding_size * sizeof(float));
            }
          }
        }
      }

    private:
        int64_t num_buckets_;
};

REGISTER_KERNEL_BUILDER(Name("KPFusedEmbeddingWithHashBucket").Device(DEVICE_CPU), KPFusedEmbeddingWithHashBucketOp);
