// ANNCFused Op registration.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "annc_fused_op.h"

namespace tensorflow {

REGISTER_OP("ANNCFused")
    // Input classification (BladeDISC-style)
    .Input("constants: Nconstants * T")         // Compile-time constants
    .Input("fixed_shapes: Nfixed * T")          // Fixed shape inputs
    .Input("dynamic_shapes: Ndynamic * T")      // Dynamic shape inputs

    // Outputs
    .Output("outputs: num_outputs * T")

    // Core attributes
    .Attr("kernel_name: string")                // Base kernel name
    .Attr("num_outputs: int >= 1")
    .Attr("output_ranks: list(int)")            // Rank for each output
    .Attr("input_ranks: list(int) = []")        // Rank for each runtime input
    .Attr("output_shapes: list(string) = []")   // Comma-separated dims, -1 uses dynamic input dim
    .Attr("kernel_arg_order: list(int) = []")   // Order of input/output memrefs passed to kernel

    // Dynamic shape support
    .Attr("dynamic_dims: list(int)")            // Global dynamic dimension indices
    .Attr("symbolic_signature: string")         // Symbolic shape signature

    // Legacy metadata kept for GraphDef compatibility. Runtime fallback is
    // intentionally not executed by ANNCFused.
    .Attr("fallback_function: func")

    // Input counts
    .Attr("Nconstants: int >= 0")
    .Attr("Nfixed: int >= 0")
    .Attr("Ndynamic: int >= 0")

    // Data type
    .Attr("T: {float} = DT_FLOAT")

    // Shared library path produced by ANNCOptimizerPass.
    .Attr("shared_lib_path: string = ''")
    .Attr("abi: string = 'mlir_ciface'")

    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // For dynamic shapes, use PartialTensorShape with unknown dims
      // The actual output shapes will be determined at runtime
      std::vector<int> output_ranks;
      TF_RETURN_IF_ERROR(c->GetAttr("output_ranks", &output_ranks));

      for (int i = 0; i < c->num_outputs(); ++i) {
        // Create unknown shape with specified rank
        shape_inference::ShapeHandle s;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
            PartialTensorShape(), &s));
        c->set_output(i, s);
      }
      return Status();
    });

// Register kernel
REGISTER_KERNEL_BUILDER(Name("ANNCFused").Device(DEVICE_CPU), ANNCFusedOp);

// Export C API symbols for dynamic loading
extern "C" {

// Return the Op name
const char* GetANNCFusedOpName() {
  return "ANNCFused";
}

// Check if the Op is registered
bool IsANNCFusedOpRegistered() {
  return true;
}

}  // extern "C"

}  // namespace tensorflow
