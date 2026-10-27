#include "annc_fused_op.h"
#include "annc_fused_op_jit.h"
#include "tensorflow/core/lib/core/errors.h"
#ifdef ANNC_ENABLE_CONSTANT_FOLDING
#include "Support/CheckpointReader.h"
#endif

namespace tensorflow {
void ANNCFusedOp::InitializeJitWeights(OpKernelContext* context) {
  std::call_once(jit_weights_once_, [&] {
    jit_identity_ = template_fingerprint_ + ":packed-rhs-v1:" + abi_;
    const int inputs = context->num_inputs();
    jit_argument_order_ = kernel_arg_order_;
    if (jit_argument_order_.empty()) {
      if (inputs == 3 && num_outputs_ == 1 && input_ranks_.empty())
        jit_argument_order_ = {2, 0, 3, 1};
      else {
        for (int i = 0; i < inputs + num_outputs_; ++i)
          jit_argument_order_.push_back(i);
      }
    }
    jit_weights_.resize(inputs);
#ifdef ANNC_ENABLE_CONSTANT_FOLDING
    const auto input_names = ReadAnncJitInputNames(
        atir_module_path_, kernel_name_, template_fingerprint_, num_constants_,
        num_fixed_, num_dynamic_);
    annc::constant_folding::RhsDataLoader loader;
    std::vector<uint8_t> checkpoint;
    // Match the static prepack admission contract; weights remain immutable
    // for the node lifetime. Checkpoint bytes are never used as packed data.
    for (int slot = 0; slot < num_constants_ + num_fixed_; ++slot) {
      const Tensor& tensor = context->input(slot);
      if (tensor.dtype() != DT_FLOAT || tensor.dims() != 2 ||
          input_names.size() != static_cast<size_t>(inputs))
        continue;
      std::string name = input_names[slot];
      const auto colon = name.rfind(':');
      if (colon != std::string::npos) {
        if (name.substr(colon) != ":0") continue;
        name.resize(colon);
      }
      if (loader.load(name, checkpoint) &&
          checkpoint.size() == tensor.TotalBytes())
        jit_weights_[slot] = tensor;
    }
#endif
    for (size_t argument = 0; argument < jit_argument_order_.size();
         ++argument) {
      int slot = jit_argument_order_[argument];
      if (slot < 0 || slot >= inputs || jit_weights_[slot].dims() != 2)
        continue;
      jit_immutable_arguments_.push_back(argument);
      jit_identity_ += ":arg=" + std::to_string(argument);
    }
  });
}

Status ANNCFusedOp::BindJitWeights(
    const std::shared_ptr<annc::jit::JitExecutable>& executable,
    std::shared_ptr<AnncJitBinding>* result) {
  {
    mutex_lock lock(jit_fastpath_mu_);
    auto found = jit_bindings_.find(executable.get());
    if (found != jit_bindings_.end() &&
        found->second->executable.lock() == executable) {
      *result = found->second;
      return OkStatus();
    }
    for (auto it = jit_bindings_.begin(); it != jit_bindings_.end();) {
      if (it->second->executable.expired())
        it = jit_bindings_.erase(it);
      else
        ++it;
    }
  }
  auto binding = std::make_shared<AnncJitBinding>();
  binding->executable = executable;
  if (!executable->packed_rhs_specs.empty())
    binding->packed.resize(jit_weights_.size());
  for (size_t i = 0; i < executable->packed_rhs_specs.size(); ++i) {
    const auto& spec = executable->packed_rhs_specs[i];
    if (spec.argument >= static_cast<int64_t>(jit_argument_order_.size()))
      return errors::Internal("Packed RHS argument out of range");
    const int slot = jit_argument_order_[spec.argument];
    if (slot < 0 || slot >= static_cast<int>(jit_weights_.size()))
      return errors::Internal("Packed RHS must refer to an immutable input");
    const Tensor& source = jit_weights_[slot];
    if (source.dims() != 2 || source.dtype() != DT_FLOAT ||
        source.dim_size(0) != spec.k || source.dim_size(1) != spec.n ||
        binding->packed[slot])
      return errors::Internal("Packed RHS input shape or slot mismatch");
    std::string error;
    auto buffer = packed_rhs_cache_.GetOrPack(
        spec, source.flat<float>().data(), executable->prepare_rhs, i, &error);
    if (!buffer) return errors::Internal(error);
    binding->packed[slot] = std::move(buffer);
  }
  {
    mutex_lock lock(jit_fastpath_mu_);
    jit_bindings_[executable.get()] = binding;
  }
  *result = std::move(binding);
  return OkStatus();
}
}  // namespace tensorflow
