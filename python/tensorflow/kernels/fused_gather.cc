#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;

REGISTER_OP("KPFusedGather")
    .Input("data: float")
    .Input("slice_input: int64")
    .Input("begin: int32")
    .Output("out_shape: int64")
    .Output("out_indices: int32")
    .Output("out_data: float")
    .SetShapeFn(shape_inference::UnknownShape);
class KPFusedGather : public OpKernel {
 public:
  explicit KPFusedGather(OpKernelConstruction* context) : OpKernel(context) { }

  void Compute(OpKernelContext* context) override {
    VLOG(2) << "Executing KPFusedGather operator";
    const Tensor& data = context->input(0);
    const Tensor& slice_input = context->input(1);
    const Tensor& begin = context->input(2);

    OP_REQUIRES(context, slice_input.dims() == 2, errors::Internal("slice_input dims must == 2"));
    OP_REQUIRES(context, data.dims() == 2, errors::Internal("indentity dims must == 2"));

    VLOG(2) << "Input indentity shape: " << data.shape().DebugString();
    VLOG(2) << "Input slice_input shape: " << slice_input.shape().DebugString();
    VLOG(2) << "Input begin value: " << begin.SummarizeValue(10);

    int32 col = begin.flat<int32>().data()[1];
    auto data_mat = data.matrix<float>();
    auto slice_input_mat = slice_input.matrix<int64>();

    VLOG(2) << "Column index from begin: " << col;

    std::vector<int64_t> unique_values;
    std::vector<int32_t> indices(slice_input.dim_size(0));
    std::unordered_map<int64_t, int32_t> value_to_index;
    int current_index = 0;
    for (int64_t i = 0; i < slice_input.dim_size(0); ++i) {
        auto it = value_to_index.find(slice_input_mat(i, col));
        if (it == value_to_index.end()) {
            value_to_index[slice_input_mat(i, col)] = current_index;
            unique_values.push_back(slice_input_mat(i, col));
            indices[i] = current_index;
            current_index++;
        } else {
            indices[i] = it->second;
        }
    }

    Tensor* out_shape = nullptr;
    Tensor* out_indices = nullptr;
    Tensor* out_data = nullptr;

    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({1}), &out_shape));
    auto output_shape = out_shape->flat<int64>();
    output_shape(0) = static_cast<int64>(unique_values.size());

    OP_REQUIRES_OK(context,
                   context->allocate_output(
                        1, TensorShape({static_cast<int32>(indices.size())}), &out_indices));
    std::memcpy(out_indices->data(), indices.data(), indices.size() * sizeof(int32_t));
    OP_REQUIRES(context, data.dim_size(1) * unique_values.size() % 12 == 0, 
                errors::Internal("cannot reshape to [-1, 12]"));
    
    std::vector<float> gather1_result;
    for (auto &indice : unique_values) {
        for (int64_t i = 0; i < data.dim_size(1); ++i) {
            gather1_result.push_back(data_mat(indice, i));
        }
    }

    OP_REQUIRES_OK(context,
                   context->allocate_output(
                        2, TensorShape({unique_values.size(), 12}), &out_data));
    auto output_data = out_data->matrix<float>();
    int cur_row = 0;
    for (auto &indice : unique_values) {
        for (int i = 0; i < 12; ++i) {
            output_data(cur_row, i) = gather1_result[12 * indice + i];
        }
        cur_row++;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("KPFusedGather").Device(DEVICE_CPU),
                        KPFusedGather);