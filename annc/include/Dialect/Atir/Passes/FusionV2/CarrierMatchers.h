#ifndef ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_CARRIERMATCHERS_H
#define ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_CARRIERMATCHERS_H

#include <cstdint>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace atir {
namespace fusionv2 {

// Reshape views: atir.unsupported{opType="Reshape"}(src, targetShape) is the
// runtime-shape residue; stablehlo.reshape(src) is static. A static
// targetShape never crosses the site boundary (kernel reads the result type).
struct ReshapeLike {
  mlir::Operation *op = nullptr;

  // Valid view for either form; empty op otherwise.
  static ReshapeLike match(mlir::Operation *op);

  explicit operator bool() const { return op != nullptr; }

  mlir::Value src() const;          // reshaped input (operand[0])
  mlir::Value targetShape() const;  // residue only; empty for stablehlo
  mlir::Value result() const { return op->getResult(0); }
};

// Gather views (TF GatherV2 axis=0): the stablehlo.torch_index_select form
// is legalized (2 operands); the GatherV2 residue takes axis as operand 2
// and only accepts a constant 0, kept out of the ABI by collectBoundary.
struct GatherLike {
  mlir::Operation *op = nullptr;

  // Valid view for either form; empty op otherwise.
  static GatherLike match(mlir::Operation *op);

  explicit operator bool() const { return op != nullptr; }

  mlir::Value params() const;   // look-up data (operand[0])
  mlir::Value indices() const;  // indices (operand[1])
  // True when the gather axis is statically 0 (kernel precondition): reads
  // dim/batch_dims attrs on the stablehlo form, the operand-2 constant on
  // the residue form (match already requires 0).
  bool axisIsZero() const;
  mlir::Value result() const { return op->getResult(0); }
};

// Slice views over stablehlo.slice / real_dynamic_slice / dynamic_slice and
// the StridedSlice residue (mask attrs stay local). Only source() and
// start() are normalized (start empty for static); mask/shrink stay
// residue-only — prefer bail-out over fake normalization.
struct SliceLike {
  mlir::Operation *op = nullptr;

  static SliceLike match(mlir::Operation *op);

  explicit operator bool() const { return op != nullptr; }

  mlir::Value source() const;  // sliced tensor (operand[0])
  mlir::Value start() const;   // start index; empty for static slice
  bool isResidue() const;      // true = atir.unsupported{StridedSlice}
  mlir::Value result() const { return op->getResult(0); }
};

// Concatenate views: stablehlo.concatenate{dimension} or the ConcatV2
// residue (trailing axis operand). inputs() drops that operand; dimension()
// returns the normalized constant axis or nullopt when unknown.
struct ConcatLike {
  mlir::Operation *op = nullptr;

  static ConcatLike match(mlir::Operation *op);

  explicit operator bool() const { return op != nullptr; }

  llvm::SmallVector<mlir::Value, 4> inputs() const;
  std::optional<int64_t> dimension() const;
  mlir::Value result() const { return op->getResult(0); }
};

// Fill views: the Fill residue keeps (dims, value) operand order, while the
// broadcast forms place the scalar first. value() is the fill scalar,
// shape() the target-shape value.
struct FillLike {
  mlir::Operation *op = nullptr;

  static FillLike match(mlir::Operation *op);

  explicit operator bool() const { return op != nullptr; }

  mlir::Value value() const;
  mlir::Value shape() const;
  mlir::Value result() const { return op->getResult(0); }
};

// Semantic groups (1:N expansion): non-legalizable TF ops expand into N
// stablehlo ops tagged annc.source_op / annc.source_node / annc.source_ports
// (resultNo, -1 unknown) by the FE; matchers only read and bail out when
// tags are missing or ambiguous.
llvm::StringRef semanticOf(mlir::Operation *op);

// Members sharing source_node and source_op; empty when untagged.
llvm::SmallVector<mlir::Operation *, 8> semanticGroup(mlir::Operation *op);

// Value producing TF output `port` in the group; empty when unproven.
std::optional<mlir::Value> groupPortValue(mlir::Operation *op, int64_t port);

// Unique views (outputs 0=values, 1=indices): the Unsupported{Unique} residue
// maps results to ports directly, 1:N sort expansions via annc.source_ports.
// input is the sole external non-constant operand (bail on ambiguity);
// indices may be empty when unconsumed.
struct UniqueLike {
  llvm::SmallVector<mlir::Operation *, 8> members;
  mlir::Value values, indices, input;
  bool residue = false;

  static UniqueLike match(mlir::Operation *op);

  explicit operator bool() const { return values && input; }
};

// DynamicPartition views (buckets in port order): the Unsupported residue
// maps results to ports; the 1:N expansion recovers ids from per-bucket
// count chains (must form 0..n-1), data/partitions from the sort inputs.
struct PartitionLike {
  llvm::SmallVector<mlir::Operation *, 8> members;
  llvm::SmallVector<mlir::Value, 8> buckets;  // in port order
  mlir::Value data, partitions;
  bool residue = false;

  static PartitionLike match(mlir::Operation *op);

  explicit operator bool() const {
    return data && partitions && !buckets.empty();
  }
  int64_t count() const { return static_cast<int64_t>(buckets.size()); }
};

}  // namespace fusionv2
}  // namespace atir

#endif  // ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_CARRIERMATCHERS_H
