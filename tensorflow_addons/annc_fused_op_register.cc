// ANNCFused Op registration.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "annc_fused_op.h"

namespace tensorflow {

REGISTER_OP("ANNCFused")
    // Input classification (BladeDISC-style)
    .Input("constants: Nconstants * Tconstants")  // Compile-time constants
    .Input("fixed_shapes: Nfixed * Tfixed")       // Fixed shape inputs
    .Input("dynamic_shapes: Ndynamic * Tdynamic") // Dynamic shape inputs

    // Outputs
    .Output("outputs: num_outputs * Toutputs")

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
    .Attr("fusion_pattern: string = ''")
    .Attr("annc_original_nodes: list(string) = []")

    // Input counts
    .Attr("Nconstants: int >= 0")
    .Attr("Nfixed: int >= 0")
    .Attr("Ndynamic: int >= 0")

    // Data types.  The legacy T attr is retained for old GraphDefs that only
    // contain float inputs/outputs; new rewrites use the per-list attributes.
    .Attr("T: {float} = DT_FLOAT")
    .Attr("Tconstants: type = DT_FLOAT")
    .Attr("Tfixed: type = DT_FLOAT")
    .Attr("Tdynamic: type = DT_FLOAT")
    .Attr("Toutputs: type = DT_FLOAT")

    // Shared library path produced by ANNCOptimizerPass.
    .Attr("shared_lib_path: string = ''")
    .Attr("abi: string = 'mlir_ciface'")
    // Set to false only when the generated kernel is known to fully overwrite
    // every output element.
    .Attr("zero_initialize_outputs: bool = true")

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
