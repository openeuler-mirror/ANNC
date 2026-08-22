// ANNCFused Op registration.

#include "annc_fused_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("ANNCFused")
    // Input classification (BladeDISC-style)
    .Input("constants: Tconstants")     // Compile-time constants
    .Input("fixed_shapes: Tfixed")      // Fixed shape inputs
    .Input("dynamic_shapes: Tdynamic")  // Dynamic shape inputs

    // Outputs
    .Output("outputs: Toutputs")

    // Core attributes
    .Attr("kernel_name: string")                // Base kernel name
    .Attr("template_fingerprint: string = ''")  // Name-independent JIT identity
    .Attr("num_outputs: int >= 1")
    .Attr("output_ranks: list(int)")      // Rank for each output
    .Attr("input_ranks: list(int) = []")  // Rank for each runtime input
    // Comma-separated dimensions; "?" uses the corresponding dynamic input
    // dimension.
    .Attr("output_shapes: list(string) = []")
    // Permutation over the combined [input memrefs..., output memrefs...] list.
    .Attr("kernel_arg_order: list(int) = []")

    // Dynamic shape support
    .Attr("dynamic_dims: list(int)")  // Dynamic output-axis indices
    // Reserved compatibility value; runtime does not interpret it.
    .Attr("symbolic_signature: string")

    // Legacy metadata kept for GraphDef compatibility. Runtime fallback is
    // intentionally not executed by ANNCFused.
    .Attr("fallback_function: func")  // Reserved; runtime does not execute it
    .Attr("fusion_pattern: string = ''")

    // Input counts
    .Attr("Nconstants: int >= 0")
    .Attr("Nfixed: int >= 0")
    .Attr("Ndynamic: int >= 0")

    // Data types.  The legacy T attr is retained for old GraphDefs that only
    // contain float inputs/outputs; new rewrites use the per-list attributes.
    .Attr("T: {float} = DT_FLOAT")
    .Attr("Tconstants: list(type) >= 0")
    .Attr("Tfixed: list(type) >= 0")
    .Attr("Tdynamic: list(type) >= 0")
    .Attr("Toutputs: list(type) >= 1")

    // Shared library path produced by ANNCOptimizerPass.
    .Attr("shared_lib_path: string = ''")
    // Fusion-only ATIR module used by synchronous runtime compilation.
    .Attr("atir_module_path: string = ''")
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
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(PartialTensorShape(), &s));
        c->set_output(i, s);
      }
      return Status();
    });

// Register kernel
REGISTER_KERNEL_BUILDER(Name("ANNCFused").Device(DEVICE_CPU), ANNCFusedOp);

// Export C API symbols for dynamic loading
extern "C" {

// Return the Op name
const char* GetANNCFusedOpName() { return "ANNCFused"; }

// Check if the Op is registered
bool IsANNCFusedOpRegistered() { return true; }

}  // extern "C"

}  // namespace tensorflow
