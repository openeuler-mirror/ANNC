//===----------------------------------------------------------------------===//
// Atir Op definitions
//===----------------------------------------------------------------------===//
#ifndef ATIR_OPS_TD
#define ATIR_OPS_TD


include "Dialect/Atir/AtirDialect.td"
include "Dialect/Atir/Interfaces/ShapeInfer.td"
include "Dialect/Atir/Interfaces/Interpret.td"
include "Dialect/Atir/AtirEnums.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

defvar Traits = [NoTerminator, NativeOpTrait<"Atir_OpTrait">];
defvar Interfaces = [DeclareOpInterfaceMethods<ShapeInferIntfc>,
                     DeclareOpInterfaceMethods<InterpretIntfc>
                    ];

def MMLike : NativeOpTrait<"Atir_MM">;
def DMALike : NativeOpTrait<"Atir_DMA">;
def ACTLike : NativeOpTrait<"Atir_ACTIVATION">;

class Atir_Op<string mnemonic, list<Trait> traits = []> :
    Op<Atir_Dialect, mnemonic, !listconcat(traits, Traits)> {
  let summary = "Astralax BaseOp";
  let results = (outs );
}

def Atir_BufferOp : Atir_Op<"buffer", Interfaces> {
  let summary = "Allocate an uninitialized buffer for intermediate computation";
  let description = [{
    A placeholder for a temporary tensor buffer.
    The actual allocation is handled by the runtime or bufferization pass.
  }];

  let arguments = (ins);
  let results = (outs Atir_Tensor:$output);
}

def Atir_NoneOp : Atir_Op<"None"> {
  let summary = "None op, return a NoneType Value";
  let description = [{ this op is used to define none type }];
  let results = (outs NoneType:$output);
  let assemblyFormat = [{ attr-dict }];
}

def Atir_LoadOp : Atir_Op<"Load", Interfaces> {
  let summary = "Load from a tensor";
  let description = [{Returns the element at the given location from within the tensor}];
  let arguments = (ins Atir_Tensor:$input,
                    I32ArrayAttr:$axes,
                    I32ArrayAttr:$start,
                    I32ArrayAttr:$size,
                    BoolAttr:$copy
                    );
  let results = (outs Atir_Tensor:$output);
}

def Atir_CustomizeOp : Atir_Op<"Customize", Interfaces> {
  let summary = "customize op";
  let description = [{
    Customize op is defined for user defined operations.
    For example, 
      `FastAttention` op would be parsed into atir.CustomizeOp.

    Since the custom op will not be optimized in most of general passes,
    convert it into the standard operations with `--graph-plugin` or 
    `--grouping-plugin` is highly recommend. 
  }];
  let arguments = (ins Variadic<Atir_Tensor>:$inputs,
                       StrAttr:$opType);
  let results = (outs Variadic<Atir_Tensor>:$output);
  let assemblyFormat = [{
    ( `(` $inputs^ `)` )?
    attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_ConstantOp : Atir_Op<"constant", Interfaces> {
  let summary = "declare or define a constant";
  let description = [{}];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let results = (outs Atir_Tensor:$data);
  let assemblyFormat = [{
    ($sym_visibility^)? $sym_name 
    `->` type($data) attr-dict
  }];
}

def Atir_VariableOp : Atir_Op<"variable", Interfaces> {
  let summary = "declare or define a variable";
  let description = [{}];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let results = (outs Atir_Tensor:$output);
  let assemblyFormat = [{
    ($sym_visibility^)? $sym_name 
    `->` type($output) attr-dict
  }];
}

def Atir_AddOp : Atir_Op<"Add", Interfaces> {
  let summary = "add op";
  let arguments = (ins
      Atir_Tensor:$output,
      Variadic<Atir_Tensor>:$inputs,
      DefaultValuedAttr<BoolAttr, "false">:$do_relu,
      DefaultValuedAttr<F32Attr, "-1.0">:$relu_limit,
      OptionalAttr<F32Attr>:$scalar
  );
  let results = (outs Atir_Tensor:$result);
}

def Atir_ConcatOp : Atir_Op<"Concat", Interfaces> {
  let summary = "Concat operator";

  let description = [{
    1.Op Introduction
    Concatenates the given sequence of seq tensors in the given dimension.
    All tensors must either have the same shape (except in the concatenating dimension) or be empty.

    2.Math formula
    output = Concat(input1, input2, axis)
           = input1[axis] + input2[axis];

    3.activation and weight
    input(act.): input tensor;

    4.attribute
    axis: the dimension along which the input tensors will be concated together.;
    do_relu: If set true, the output will be activated via the ReLU function after the calculation is complete.
    relu_limit: If set -1.0, it means that there is no upper limit and the output will only be affected by the ReLU function.;
    round_mode: This determines how values are rounded during the conversion from higher precision to lower precision.;

  }];

  let arguments = (
    ins Atir_Tensor:$output,
    Variadic<Atir_Tensor>:$inputs,
    DefaultValuedAttr<I32Attr, "1">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F32Attr, "-1.0">:$relu_limit,
    DefaultValuedAttr<I32Attr, "0">:$round_mode,
    OptionalAttr<BoolAttr>:$is_view
  );

  let results = (outs Atir_Tensor:$result);
}

def Atir_MatMulOp : Atir_Op<"MatMul", Interfaces> {
  let summary = "matmul op";
  let arguments = (
    ins Atir_Tensor:$C, 
    Atir_Tensor:$lhs,
    Atir_Tensor:$rhs,
    Optional<Atir_Tensor>:$bias,
    BoolAttr:$withBias,
    DefaultValuedAttr<BoolAttr, "false">:$right_transpose,
    DefaultValuedAttr<BoolAttr, "false">:$left_transpose,
    DefaultValuedAttr<BoolAttr, "false">:$output_transpose,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F32Attr, "-1.0">:$relu_limit,
    OptionalAttr<I32Attr>:$m_start,
    OptionalAttr<I32Attr>:$n_start,
    OptionalAttr<I32Attr>:$k_start,
    OptionalAttr<I32Attr>:$m_size,
    OptionalAttr<I32Attr>:$n_size,
    OptionalAttr<I32Attr>:$k_size
  );
  let results = (outs Atir_Tensor:$output);
}

def Atir_ReluOp : Atir_Op<"Relu"> {
  let summary = "relu op";
  let arguments = (
      ins Atir_Tensor:$output,
      Atir_Tensor:$input,
      DefaultValuedAttr<F32Attr, "-1.0">:$relu_limit
  );
  let results = (outs Atir_Tensor:$result);
}

def Atir_ForOp : Atir_Op<"for"> {
  let summary = "for operation";
  let description = [{
    %0 = atir.alloc()
    // concat
    atir.for (i = 0; i < n; i+=s) {
      %in = extract() {[0, %i, 0 , 0] [?, s, ?, ?]}
      %1 = conv(%in)
      %2 = conv(%1)
      insert(%0, %2)
    }

    // reduce
    %0 = atir.alloc()
    atir.for (i = 0; i < n; i+=s; post_operation = ['reduce_sum'], iter_args = [%0]) {
      %in = extract() {[0, %i, 0 , 0] [?, s, ?, ?]}
      %1 = conv(%in)
      %2 = conv(%1)
      return %2
    }

    %0 = atir.alloc()
    %1 = atir.alloc()
    axt.for (i in [])
    atir.for (i = 0; i < n; i+=s; post_operation = ['reduce_sum'], iter_args = [%0, %1], parallel) {
      %in = extract() {[0, %i, 0 , 0] [?, s, ?, ?]}
      %1 = conv(%in)
      %2 = conv(%in)
      return %1, %2
    }

    %start_in = arith.Constant{[0, 7]}
    %step_in = arith.Constant{[9, 8]}
    %start_out = arith.Constant([0, 4])
    %step_out = arith.Constant([4, 4])
    atir.for %arg_start_in, %arg_step_in, %arg_start_out, %arg_step_out in zip(%start_in, %step_in, %start_out, %step_out) {
      %in = extract() {[0, %arg_start_in, 0, 0], [?, %arg_step_in, ?, ?]}
      inset() {[0, %arg_start_out, 0, 0], [?, %arg_step_out, ?, ?]}
    }
  }];
  let arguments = (ins I64Attr:$idxCnt,
                       Variadic<AnyType>:$initArgs,
                       /* arg order:
                       [2]: iter idx0 constant
                       [3]: iter idx1 constant
                       ...
                       [...] iter args
                       */
                       OptionalAttr<I64Attr>:$start,
                       OptionalAttr<I64Attr>:$end,
                       OptionalAttr<I64Attr>:$step,
                       OptionalAttr<I64Attr>:$axis,
                       /* reduce_sum/min/max, concat */
                       OptionalAttr<StrAttr>:$postOp,
                       OptionalAttr<BoolAttr>:$parallel);
  let results = (outs Variadic<Atir_Tensor>:$results);
  let regions = (region SizedRegion<1>:$region);

  let extraClassDeclaration = [{
    ::mlir::MutableArrayRef<mlir::BlockArgument> getInductionVars() {
      return getRegion().getArguments();
    }
  }];
  
  let builders = [
    OpBuilder<(ins CArg<"int64_t">:$idxCnt,
      CArg<"::mlir::ValueRange", "std::nullopt">:$iterArgs,
      CArg<"::mlir::IntegerAttr", "{}">:$axis,
      CArg<"::mlir::StringAttr", "{}">:$postOp,
      CArg<"::mlir::BoolAttr", "{}">:$parallel)>,
    OpBuilder<(ins CArg<"int64_t">:$start,
      CArg<"int64_t">:$end,
      CArg<"int64_t">:$step,
      CArg<"::mlir::ValueRange", "std::nullopt">:$iterArgs,
      CArg<"::mlir::IntegerAttr", "{}">:$axis,
      CArg<"::mlir::StringAttr", "{}">:$postOp,
      CArg<"::mlir::BoolAttr", "{}">:$parallel)>,
  ];
  let hasCustomAssemblyFormat = 1;
}

def Atir_IfOp : Atir_Op<"if"> {
  let summary = "if-then-else operation";
  let description = [{}];
  let arguments = (ins AnyType:$lhs,
                       AnyType:$rhs,
                       StrAttr:$LogicOperation,
                       OptionalAttr<F32Attr>:$constValue
                       );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let builders = [];
  // let hasCustomAssemblyFormat = 1;
}

def Atir_SwitchCaseOp : Atir_Op<"switch", [SingleBlockImplicitTerminator<"atir::ReturnOp">]> {
  let summary = "switch-case operation on an index argument";
  let arguments = (ins Index:$arg, DenseI64ArrayAttr:$cases);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$defaultRegion,
                        VariadicRegion<SizedRegion<1>>:$caseRegions);

  let assemblyFormat = [{
    $arg attr-dict (`->` type($results)^)?
    custom<SwitchCases>($cases, $caseRegions) `\n`
    `` `default` $defaultRegion
  }];

  let extraClassDeclaration = [{
    /// Get the number of cases.
    unsigned getNumCases();

    /// Get the default region body.
    ::mlir::Block &getDefaultBlock();

    /// Get the body of a case region.
    ::mlir::Block &getCaseBlock(unsigned idx);
  }];
  // let builders = [
  //   OpBuilder<(ins CArg<"::mlir::Value">:$arg,
  //     CArg<"::llvm::SmallVector<int64_t>", "{}">:$cases,
  //     CArg<"::mlir::IntegerAttr", "{}">:$axis,
  //     CArg<"::mlir::StringAttr", "{}">:$postOp,
  //     CArg<"::mlir::BoolAttr", "{}">:$parallel)>]
}

def Atir_ParallelOp : Atir_Op<"Parallel"> {
  let summary = "parallel group";
  let arguments = (ins Variadic<Atir_Tensor>:$inputs);
  let results = (outs Variadic<Atir_Tensor>:$outputs);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
    ( `(` $inputs^ `)` )?
    attr-dict `:` functional-type(operands, results)
    $region
  }];
}

def Atir_ReturnOp : Atir_Op<"return", !listconcat([Terminator, ParentOneOf<["ForOp", "ParallelOp", "SwitchCaseOp"]>], Interfaces)> {
  let summary = "loop yield and termination operation";
  let description = [{}];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];
}

def Atir_ExpandDimsOp : Atir_Op<"ExpandDims", !listconcat([Pure], Interfaces)> {
  let summary = "Inserts a dimension of 1 into a tensor's shape";
  let description = [{
    Given a tensor input, this operation inserts a dimension of size 1 at the dimension index 
    specified by axis. The axis is zero-based; if a negative value is specified, the index is 
    counted from the end of the shape.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$input,
    I32Attr:$axis
  );
  let results = (outs Atir_Tensor:$result);
  let assemblyFormat = [{
    $output `,` $input attr-dict `:` type($output) `,` type($input) `->` type($result)
  }];
}

def Atir_IdentityOp : Atir_Op<"Identity", !listconcat([Pure, SameOperandsAndResultElementType,
    SameOperandsAndResultShape], Interfaces)> {
  let summary = "Identity operation";
  let description = [{
    Returns the input tensor as-is. This op is often used for 
    naming, control dependencies, or intermediate value anchoring.
  }];
  let arguments = (ins Atir_Tensor:$output, Atir_Tensor:$input);
  let results = (outs Atir_Tensor:$result);
  let assemblyFormat = [{
    $output `,` $input attr-dict `:` type($output) `,` type($input) `->` type($result)
  }];
  let hasFolder = 1;
}

def Atir_ShapeOp : Atir_Op<"Shape", !listconcat([Pure], Interfaces)> {
  let summary = "Returns the input shape information.";
  let description = [{
    Given a tensor input, this operation generates a one-dimensional tensor 
    representing the shape information of the input.
  }];
  let arguments = (ins
    Atir_Tensor:$input
  );
  let results = (outs 
    Atir_Tensor:$output
  );
  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def Atir_NotEqualOp : Atir_Op<"NotEqual", !listconcat([NoMemoryEffect,
    Elementwise, SameOperandsAndResultShape], Interfaces)> {
  let summary = "Returns (input1 != input2) element-wise";
  let description = [{
    Returns the truth value of (input1 != input2) element-wise.
    Supports broadcasting between input1 and input2.
  }];
  let arguments = (ins
    Atir_Tensor:$input1,
    Atir_Tensor:$input2
  );
  let results = (outs
    Atir_Tensor:$output
  );
  let assemblyFormat = [{
    $input1 `,` $input2 attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_CastOp: Atir_Op<"Cast", !listconcat([Pure, SameOperandsAndResultShape], Interfaces)>{
  let summary = "Cast input to a new type";
  let description = [{
    Casts a tensor from one data type to another.
    The shape of the tensor remains unchanged.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$input
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $input attr-dict `:` type($output) `,` type($input) `->` type($result)
  }];
}

def Atir_WhereOp : Atir_Op<"Where", !listconcat([Pure], Interfaces)> {
  let summary = "Returns elements from x or y depending on condition.";
  let description = [{
    Selects elements from `x` or `y` depending on `condition`.
    When `x` and `y` are provided: output[i] = condition[i] ? x[i] : y[i]
    When only `condition` is provided: returns indices where condition is true.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Variadic<Atir_Tensor>:$inputs
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` `(` $inputs `)` attr-dict `:` type($output) `,` type($inputs) `->` type($result)
  }];
}

def Atir_StridedSliceOp : Atir_Op<"StridedSlice", !listconcat([Pure], Interfaces)> {
  let summary = "StridedSlice operation";
  let description = [{
    TensorFlow-compatible StridedSlice: 1-D tensor operands `begin`, `end`, and
    `strides` (same length as the sparse slice spec, less than 32), plus scalar
    bitmask attributes `beginMask`, `endMask`, `ellipsisMask`, `newAxisMask`, and
    `shrinkAxisMask`. At most one bit may be set in `ellipsisMask`; if none is
    set, an ellipsis is implied at the end of the spec (matching TensorFlow).
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$input,
    Atir_Tensor:$begin,
    Atir_Tensor:$end,
    Atir_Tensor:$strides,
    DefaultValuedAttr<I32Attr, "0">:$beginMask,
    DefaultValuedAttr<I32Attr, "0">:$endMask,
    DefaultValuedAttr<I32Attr, "0">:$ellipsisMask,
    DefaultValuedAttr<I32Attr, "0">:$newAxisMask,
    DefaultValuedAttr<I32Attr, "0">:$shrinkAxisMask
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $input `,` $begin `,` $end `,` $strides attr-dict `:` type($output) `,` type($input) `,` type($begin) `,` type($end) `,` type($strides) `->` type($result)
  }];
}

def Atir_PackOp : Atir_Op<"Pack", !listconcat([Pure, SameOperandsShape], Interfaces)> {
  let summary = "Pack tensors along a new axis";
  let description = [{
    Packs the `N` tensors in `values` along the `axis` dimension.
    Given a list of tensors of shape `(A, B, C)`;
    if `axis == 0` then the `output` tensor will have shape `(N, A, B, C)`.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Variadic<Atir_Tensor>:$inputs,
    I64Attr:$axis
  );
  let results = (outs Atir_Tensor:$result);
  let assemblyFormat = [{
    $output `,` `(` $inputs `)` `axis` `=` $axis attr-dict `:` 
    type($output) `,` type($inputs) `->` type($result)
  }];
}

def Atir_GatherNdOp : Atir_Op<"GatherNd", !listconcat([Pure], Interfaces)> {
  let summary = "Gather slices from params using nd indices";
  let description = [{
    Gathers slices from `params` into a Tensor with shape specified by 
    `indices`. `indices` is a K-dimensional integer tensor, best thought of 
    as a (K-1)-dimensional tensor of indices into `params`, where each 
    element defines a slice of `params`.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$values,
    Atir_Tensor:$indices
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $values `,` $indices attr-dict `:` type($output) `,` type($values) `,` type($indices) `->` type($result)
  }];
}

def Atir_StringToHashBucketFastOp : Atir_Op<"StringToHashBucketFast",
    !listconcat([Pure, SameOperandsAndResultShape], Interfaces)> {
  let summary = "Map strings to hash bucket indices.";
  let description = [{
    Maps each string in `input` to an integer hash modulo `numBuckets`.
    Hash function is deterministic within a run but not guaranteed across versions.
  }];
  let arguments = (ins
      Atir_Tensor:$output,
      Atir_Tensor:$input,
      I64Attr:$numBuckets
    );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $input attr-dict `:` type($output) `,` type($input) `->` type($result)
  }];
}

def Atir_SparseReshapeOp : Atir_Op<"SparseReshape", !listconcat([Pure], Interfaces)> {
  let summary = "Reshape a SparseTensor to a new dense shape.";
  let description = [{
    This operation reshapes a sparse tensor represented by indices and values 
    to a new dense shape. The non-zero elements remain the same, only their 
    positions change according to the new shape.
  }];
  let arguments = (ins
    Atir_Tensor:$indices,
    Atir_Tensor:$inputShape,
    Atir_Tensor:$newShape
  );
  let results = (outs
    Atir_Tensor:$outputIndices,
    Atir_Tensor:$outputShape
  );
  let assemblyFormat = [{
    $indices `,` $inputShape `,` $newShape attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_SliceOp : Atir_Op<"Slice", Interfaces> {
  let summary = "Extracts slices of the tensor";
  let description = [{
    TF `Slice`: `input`, 1D `begin`, 1D `size` (stride 1).
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$input,
    Atir_Tensor:$begin,
    Atir_Tensor:$size
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $input `,` $begin `,` $size attr-dict `:` type($output) `,` type($input) `,` type($begin) `,` type($size) `->` type($result)
  }];
}

def Atir_GatherOp : Atir_Op<"Gather", Interfaces> {
  let summary = "Gather slices from params axis according to indices"; 
  let description = [{
    Gathers slices from `params` along the `axis` dimension according to `indices`. 
    Produces an output tensor with shape:
    params.shape[:axis] + indices.shape[batch_dims:] + params.shape[axis + 1:]
    The `batch_dims` attribute specifies how many leading dimensions of `indices`
    are batch dimensions that should not be gathered over.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$params,
    Atir_Tensor:$indices,
    Atir_Tensor:$axis,
    DefaultValuedAttr<I32Attr, "0">:$batch_dims
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $params `,` $indices `,` $axis attr-dict `:` type($output) `,` type($params) `,` type($indices) `,` type($axis) `->` type($result)
  }];
}

def Atir_ProdOp : Atir_Op<"Prod", Interfaces> {
  let summary = "Compute the product of tensor elements over specified axes"; 
  let description = [{
    Computes the product of elements across dimensions of a tensor.
    Reduces `input` along the dimensions specified in `indices`. 
    If `keep_dims` is true, the reduced dimensions are retained with size 1.
    Otherwise, the reduced dimensions are removed.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$input,
    Atir_Tensor:$indices,
    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $input `,` $indices attr-dict `:` type($output) `,` type($input) `,` type($indices) `->` type($result)
  }];
}

def Atir_SumOp : Atir_Op<"Sum", Interfaces> {
  let summary = "Compute the sum of tensor elements over specified axes";
  let description = [{
    Computes the sum of elements across dimensions of a tensor.
    Reduces `input` along the dimensions specified in `indices`.
    If `keep_dims` is true, the reduced dimensions are retained with size 1.
    Otherwise, the reduced dimensions are removed.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$input,
    Atir_Tensor:$indices,
    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $input `,` $indices attr-dict `:` type($output) `,` type($input) `,` type($indices) `->` type($result)
  }];
}

def Atir_GreaterEqualOp : Atir_Op<"GreaterEqual", Interfaces> {
  let summary = "Element-wise greater-than-or-equal comparison";
  let description = [{
    Computes element-wise greater-than-or-equal comparison of `lhs` and `rhs`.
    Returns a boolean tensor indicating where `lhs >= rhs`.
  }];
  let arguments = (ins
    Atir_Tensor:$lhs,
    Atir_Tensor:$rhs
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_SparseFillEmptyRowsOp : Atir_Op<"SparseFillEmptyRows", 
    !listconcat([Pure], Interfaces)> {
  let summary = "Fills empty rows in a sparse tensor with a default value.";
  let description = [{
    This operation fills missing rows in a sparse tensor representation 
    with a single entry at [row, 0, ..., 0] containing the default value.
    The input sparse tensor is represented by (indices, values, denseShape).
    The output has the same denseShape but with filled empty rows in sorted order.
    Returns:
    - outputIndices: The filled sparse tensor indices
    - outputValues: The filled sparse tensor values  
    - emptyRowIndicator: Boolean vector indicating which rows were empty
    - reverseIndexMap: Maps input indices to output indices for backprop
  }];
  let arguments = (ins
    Atir_Tensor:$indices,           // [N, rank] int64
    Atir_Tensor:$values,                  // [N] any type T
    Atir_Tensor:$denseShape,       // [rank] int64
    Atir_Tensor:$defaultValue              // scalar of type T
  );
  let results = (outs
    Atir_Tensor:$outputIndices,    // [N_full, rank] int64
    Atir_Tensor:$outputValues,           // [N_full] same type as values
    Atir_Tensor:$emptyRowIndicator,// [dense_shape[0]] bool
    Atir_Tensor:$reverseIndexMap  // [N] int64
  );
  let assemblyFormat = [{
    $indices `,` $values `,` $denseShape `,` $defaultValue attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_VarHandleOp : Atir_Op<"VarHandleOp", [NoMemoryEffect, Pure]> {
  let summary = "Returns a handle to a variable resource";
  let description = [{
    Creates a handle to a variable resource. The handle can be used with 
    ReadVariableOp and AssignVariableOp to read from and write to the variable.
  }];
  let arguments = (ins
    StrAttr:$container,
    StrAttr:$shared_name,
    TypeAttr:$dtype,
    I64ArrayAttr:$shape
  );
  // let results = (outs
  //  Atir_ResourceType:$resource_handle
  // );
}

def Atir_UniqueOp : Atir_Op<"Unique", !listconcat([Pure], Interfaces)> {
  let summary = "Finds unique elements in a 1-D tensor and returns them with indices.";
  let description = [{
    This operation returns a tensor `y` containing all of the unique elements of `x`
    sorted in the same order that they occur in `x`. It also returns another tensor
    `idx` the same size as `x` that contains the index of each value of `x` in the
    unique output `y`. 
  }];
  let arguments = (ins
    Atir_Tensor:$x
  );
  let results = (outs
    Atir_Tensor:$y,
    Atir_Tensor:$idx
  );
  let assemblyFormat = [{
    $x attr-dict `:` functional-type(operands, results)
  }]; 
}

def Atir_ResourceGatherOp : Atir_Op<"ResourceGather", !listconcat([Pure], Interfaces)> {
  let summary = "Gather elements from resource handle by indices";
  let description = [{
    Gathers data from a resource according to given indices.
    Supports batched gathering with multiple index dimensions.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    AnyType:$resource,
    Variadic<AnyType>:$indices,
    DefaultValuedAttr<BoolAttr, "false">:$validateIndices,
    DefaultValuedAttr<I64Attr, "0">:$batchDims
  );
  let results = (outs Atir_Tensor:$result);
  let assemblyFormat = [{
    $output `,` $resource `,` $indices attr-dict `:` type($output) `,` type($resource) `,` type($indices) `->` type($result)
  }];
}

def Atir_SparseSegmentSumOp : Atir_Op<"SparseSegmentSum", !listconcat([Pure], Interfaces)> {
  let summary = "Computes sums of sparse segments of a tensor.";
  let description = [{
    Computes the sum of rows in `data` selected by `indices`, grouped by `segment_ids`.
    - `data`: Input tensor of shape `[D0, D1, ..., Dn]`
    - `indices`: 1D integer tensor specifying which rows of `data` to use
    - `segment_ids`: 1D integer tensor (same length as `indices`), must be sorted 
      and start from 0 with no gaps. Each value assigns the corresponding indexed row to a segment.
    - `num_segments`: scalar integer tensor. If the constant value is `<= 0`, the leading output
      dimension is inferred as `max(segment_ids) + 1` from `segment_ids` data when available.
      If `> 0`, it is the explicit number of segment rows (padded with zeros), and must be
      at least `max(segment_ids) + 1`.
    Output shape: `[num_segments] + data.shape[1:]` (inferred or explicit as above).
    Output element type matches `data`.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor: $input,
    Atir_Tensor: $indices,
    Atir_Tensor: $segmentIds,
    Atir_Tensor:$numSegments
  );
  let results = (outs
    Atir_Tensor: $result
  );
  let assemblyFormat = [{
    $output `,` $input `,` $indices `,` $segmentIds `,` $numSegments attr-dict `:` type($output) `,` type($input) `,` type($indices) `,` type($segmentIds) `,` type($numSegments) `->` type($result)
  }];      
}

def Atir_SparseSegmentMinOp : Atir_Op<"SparseSegmentMin", !listconcat([Pure], Interfaces)> {
  let summary = "Computes minima of sparse segments of a tensor.";
  let description = [{
    Computes the element-wise minimum of rows in `data` selected by `indices`, grouped by `segment_ids`.
    - `data`: Input tensor of shape `[D0, D1, ..., Dn]`
    - `indices`: 1D integer tensor specifying which rows of `data` to use
    - `segment_ids`: 1D integer tensor (same length as `indices`), sorted, starting from 0 with no gaps.
    - `num_segments`: scalar integer tensor; `<= 0` means infer row count from `segment_ids` (`max + 1`);
      `> 0` is an explicit segment count (padding), at least `max(segment_ids) + 1`.
    Output shape: `[num_segments] + data.shape[1:]`. Empty segments yield `+inf` for float inputs (same convention as `UnsortedSegmentMin`).
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor: $input,
    Atir_Tensor: $indices,
    Atir_Tensor: $segmentIds,
    Atir_Tensor: $numSegments
  );
  let results = (outs
    Atir_Tensor: $result
  );
  let assemblyFormat = [{
    $output `,` $input `,` $indices `,` $segmentIds `,` $numSegments attr-dict `:` type($output) `,` type($input) `,` type($indices) `,` type($segmentIds) `,` type($numSegments) `->` type($result)
  }];
}

def Atir_TileOp : Atir_Op<"Tile", Interfaces> {
  let summary = "Tile a tensor along specified multiples";
  let description = [{
    Constructs a tensor by tiling `input` by `multiples`.
    The output tensor has the same rank as `input`, and each dimension 
    of the output is `multiples[i] * input.shape[i]`.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$input,
    Atir_Tensor:$multiples
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $input `,` $multiples attr-dict `:` type($output) `,` type($input) `,` type($multiples) `->` type($result)
  }];
}

def Atir_ZerosLikeOp : Atir_Op<"ZerosLike", [Pure, SameOperandsAndResultShape]> {
  let summary = "Returns a tensor of zeros with the same shape and type as input";
  let description = [{
    The ZerosLike operation returns a new tensor where all elements are zero,
    matching the shape and element type of the input operand. This is useful
    for initializing tensors in computations.
  }];
  let arguments = (ins
    Atir_Tensor:$input
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_SelectOp : Atir_Op<"Select"> {
  let summary = "Select elements from `input1` or `input2` depending on `condition`";
  let description = [{
    Selects elements from `input1` or `input2` based on the values of `condition`.
    For each element, if the corresponding element in `condition` is true,
    the element from `input2` is selected, otherwise the element from `input1` is selected.
  }];
  let arguments = (ins
    Atir_Tensor:$condition,
    Atir_Tensor:$input1,
    Atir_Tensor:$input2
  );
  let results = (outs
    Atir_Tensor:$output
  );
  let assemblyFormat = [{
    $condition `,` $input1 `,` $input2 attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_ConcatV2Op : Atir_Op<"ConcatV2", Interfaces> {
  let summary = "Concatenate tensors along an axis";
  let description = [{
    Concatenates a list of tensors along dimension `axis`.
    All tensors must have the same rank and same size in all dimensions
    except `axis`.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Variadic<Atir_Tensor>:$values,
    Atir_Tensor:$axis
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_SizeOp : Atir_Op<"Size", !listconcat([Pure], Interfaces)> {
  let summary = "Return the size of tensor.";
  let description = [{
    Computes the total number of elements in the input tensor.
  }];
  let arguments = (ins Atir_Tensor:$input);
  let results = (outs Atir_Tensor:$output);  // The output must be a scalar and an integer.
  let hasVerifier = 1;
  let hasFolder = 1;
  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_FloorModOp : Atir_Op<"FloorMod", !listconcat([Pure,
    ResultsBroadcastableShape, SameOperandsAndResultShape], Interfaces)> {
  let summary = "Element-wise remainder of division";
  let description = [{
    Returns element-wise remainder of division: `floor(x / y) * y + mod = x`.
    Matches Python's `%` operator.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$x,
    Atir_Tensor:$y
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $x `,` $y attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_FloorDivOp : Atir_Op<"FloorDiv", !listconcat([Pure, SameOperandsAndResultShape,
    ResultsBroadcastableShape], Interfaces)>{
  let summary = "Element-wise floor division: floor(x / y). Matches Python // semantics.";
  let description = [{
    Returns element-wise floor division: `x / y`.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$x,
    Atir_Tensor:$y
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $x `,` $y attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_RangeOp : Atir_Op<"Range", !listconcat([Pure], Interfaces)> {
  let summary = "Creates a sequence of numbers";
  let description = [{
    Creates a sequence of numbers from `start` to `limit` with step `delta`.
  }];
  let arguments = (ins  // The input must be a scalar.
    Atir_Tensor:$output,
    Atir_Tensor:$start,
    Atir_Tensor:$limit,
    Atir_Tensor:$delta
  );
  let results = (outs  // The output must be a one-dimensional tensor.
    Atir_Tensor:$result
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    $output `,` $start `,` $limit `,` $delta attr-dict `:` type($output) `,` type($start) `,` type($limit) `,` type($delta) `->` type($result)
  }];
}

def Atir_DynamicPartitionOp : Atir_Op<"DynamicPartition", !listconcat([Pure], Interfaces)> {
  let summary = "Partitions data into numPartitions tensors using indices from partitions";
  let description = [{
    Partitions data into numPartitions tensors according to the indices in partitions.
  }];
  let arguments = (ins
    Atir_Tensor:$data,
    Atir_Tensor:$partitions,
    I32Attr:$numPartitions
  );
  let results = (outs
    Variadic<Atir_Tensor>:$outputs
  );
  let assemblyFormat = [{
    $data `,` $partitions attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_ParallelDynamicStitchOp : Atir_Op<"ParallelDynamicStitch",
    !listconcat([Pure, SameVariadicOperandSize], Interfaces)> {
  let summary = "Interleave values from multiple tensors into a single tensor using parallel indices";
  let description = [{
    Interleaves values from multiple tensors into a single tensor using indices arrays.
    This is the inverse of DynamicPartition. Works in parallel for better performance.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Variadic<Atir_Tensor>:$indices,
    Variadic<Atir_Tensor>:$data
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_SparseSegmentMeanOp : Atir_Op<"SparseSegmentMean", !listconcat([Pure], Interfaces)> {
  let summary = "Computes the mean along sparse segments of a tensor";
  let description = [{
    Computes the mean along sparse segments of a tensor.
    Example:
      data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
      indices = [0, 1, 2, 3]
      segment_ids = [0, 0, 1, 1]
      
      output[0] = mean(data[0], data[1]) = [2.5, 3.5, 4.5]
      output[1] = mean(data[2], data[3]) = [8.5, 9.5, 10.5]
    - `num_segments`: scalar integer tensor. If the constant value is `<= 0`, the leading output
      dimension is inferred as `max(segment_ids) + 1` from `segment_ids` when available.
      If `> 0`, it is the explicit segment count (extra segments have mean 0 / no contribution).
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$data,
    Atir_Tensor:$indices,
    Atir_Tensor:$segmentIds,
    Atir_Tensor:$numSegments
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $data `,` $indices `,` $segmentIds `,` $numSegments attr-dict `:` type($output) `,` type($data) `,` type($indices) `,` type($segmentIds) `,` type($numSegments) `->` type($result)
  }];
}

def Atir_SubOp : Atir_Op<"Sub", !listconcat([Pure, Commutative,
    ResultsBroadcastableShape, SameOperandsAndResultShape], Interfaces)> {
  let summary = "Element-wise subtraction: x - y.";
  let description = [{
    Computes per-element subtraction of two tensors with standard broadcasting semantics.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$x,
    Atir_Tensor:$y
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $x `,` $y attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_MulOp : Atir_Op<"Mul", !listconcat([Pure], Interfaces)> {
  let summary = "Element-wise multiplication with broadcasting";
  let description = [{
    Computes output = lhs * rhs element-wise.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$x,
    Atir_Tensor:$y
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $x `,` $y attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_LessOp : Atir_Op<"Less", !listconcat([Pure, ResultsBroadcastableShape], Interfaces)> {
  let summary = "Element-wise less-than comparison operation (lhs < rhs).";
  let description = [{
    Performs element-wise less-than comparison of two tensors with broadcast support.
  }];
  let arguments = (ins
    Atir_Tensor:$lhs,
    Atir_Tensor:$rhs
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_FillOp : Atir_Op<"Fill", !listconcat([Pure], Interfaces)> {
  let summary = "Create tensor of shape filled with scalar value";
  let description = [{
    Generates a tensor with a specified shape (from shape input) where all elements 
    are set to the given scalar value (from value input).
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$shapeInput,
    Atir_Tensor:$valueInput
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    $output `,` $shapeInput `,` $valueInput attr-dict `:` type($output) `,` type($shapeInput) `,` type($valueInput) `->` type($result)
  }];
}

def Atir_GreaterOp : Atir_Op<"Greater", !listconcat([Pure, ResultsBroadcastableShape], Interfaces)> {
  let summary = "Element-wise greater-than comparison operation (lhs > rhs) ";
  let description = [{
    Performs element-wise greater-than comparison of two tensors with broadcast support.
  }];
  let arguments = (ins
    Atir_Tensor:$lhs,
    Atir_Tensor:$rhs
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_RealDivOp : Atir_Op<"RealDiv", !listconcat([Pure], Interfaces)> {
  let summary = "For the real number type, element-wise returns x / y.";
  let description = [{
    If `x` and `y` are real numbers, this will return the result of floating-point division.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$x,
    Atir_Tensor:$y
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $x `,` $y attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_SparseToDenseOp : Atir_Op<"SparseToDense", !listconcat([Pure], Interfaces)> {
  let summary = "Converts sparse representations into dense tensors.";
  let description = [{
    Converts sparse representation (sparseIndice + sparseValues) to a dense tensor with specified shape.
    The output tensor is initialized with defaultValue, then filled with values at the positions specified by indices.
    The index should be sorted in lexicographical order, and it should not contain any duplicate items. 
    If `validateIndices` is true, these properties will be checked during the execution process.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$sparseIndice,
    Atir_Tensor:$outputShape,
    Atir_Tensor:$sparseValue,
    Atir_Tensor:$defaultValue,
    DefaultValuedOptionalAttr<BoolAttr, "true">:$validateIndices
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $sparseIndice `,` $outputShape `,` $sparseValue `,` $defaultValue attr-dict `:` type($output) `,` type($sparseIndice) `,` type($outputShape) `,` type($sparseValue) `,` type($defaultValue) `->` type($result)
  }];
}

def Atir_BatchMatMulOp : Atir_Op<"BatchMatMul", !listconcat([Pure], Interfaces)> {
  let summary = "Performs batch multiplication on slices of two tensors.";
  let description = [{
    Performs batch matrix multiplication on high-dimensional tensors. For each batch, 
    the matrix multiplication is computed on the submatrices, where the last two dimensions of A/B may be transposed or not,
    which is controlled by the transposeA/transposeB attributes.
    The batch dimensions are handled according to TensorFlow's batch matrix multiply broadcast rules.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$A,
    Atir_Tensor:$B,
    DefaultValuedAttr<BoolAttr, "false">:$transposeA,
    DefaultValuedAttr<BoolAttr, "false">:$transposeB
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    $output `,` $A `,` $B attr-dict `:` type($output) `,` type($A) `,` type($B) `->` type($result)
  }];
}

def Atir_RsqrtOp : Atir_Op<"Rsqrt", [Pure, SameOperandsAndResultType, Elementwise]> {
  let summary = "Computes reciprocal of square root element-wise.";
  let description = [{
    Performs element-wise reciprocal square root on tensor `x`, producing tensor `y`.
    For each element in the input: `y = 1 / sqrt(x)`.
  }];
  let arguments = (ins
    Atir_Tensor:$x
  );
  let results = (outs
    Atir_Tensor:$y
  );
  let assemblyFormat = [{
    $x attr-dict `:` type($x)
  }];
}

def Atir_TopKOp : Atir_Op<"TopK", !listconcat([Pure], Interfaces)> {
  let summary = "Find the values and indices of the `k` largest elements along the last dimension.";
  let description = [{
    Find the values and indices of the `k` largest elements along the last dimension.
  }];
  let arguments = (ins
    Atir_Tensor:$input,
    Atir_Tensor:$k,
    DefaultValuedOptionalAttr<BoolAttr, "true">:$sorted
  );
  let results = (outs
    Atir_Tensor:$values,
    Atir_Tensor:$indices
  );
  let hasVerifier = 1;
  let assemblyFormat=[{
    $input `,` $k attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_UnsortedSegmentMinOp : Atir_Op<"UnsortedSegmentMin", !listconcat([Pure], Interfaces)> {
  let summary = "Computes the minimum value in each segment of the tensor.";
  let description = [{
    Aggregates elements of the input tensor into segments specified by segmentIds (unsorted),
    and computes the minimum value for each segment. The output tensor has shape [num_segments] + input.shape[1:].
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$input,
    Atir_Tensor:$segmentIds,
    Atir_Tensor:$numSegments
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat=[{
    $output `,` $input `,` $segmentIds `,` $numSegments attr-dict `:` type($output) `,` type($input) `,` type($segmentIds) `,` type($numSegments) `->` type($result)
  }];
}

def Atir_MinimumOp : Atir_Op<"Minimum", !listconcat([Pure,
    ResultsBroadcastableShape], Interfaces)> {
  let summary = "Returns the minimum value of x and y element-wise.";
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$x,
    Atir_Tensor:$y
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat=[{
    $output `,` $x `,` $y attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_TensorScatterUpdateOp : Atir_Op<"TensorScatterUpdate", !listconcat([Pure, ResultsBroadcastableShape], Interfaces)> {
  let summary = "Scatter update tensor values";
  let description = [{
    Updates values of the input tensor at specified indices with values from the updates tensor,
    and returns a new tensor (non-in-place update). 
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$input,
    Atir_Tensor:$indices,
    Atir_Tensor:$updates
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat=[{
    $output `,` $input `,` $indices `,` $updates attr-dict `:` type($output) `,` type($input) `,` type($indices) `,` type($updates) `->` type($result)
  }];
}

def Atir_PadOp : Atir_Op<"Pad", !listconcat([Pure], Interfaces)> {
  let summary = "Pad a tensor with specified value";
  let description = [{
    Pads a tensor according to the paddings tensor.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$input,
    Atir_Tensor:$paddings,
    Atir_Tensor:$value
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    $output `,` $input `,` $paddings `,` $value attr-dict `:` type($output) `,` type($input) `,` type($paddings) `,` type($value) `->` type($result)
  }];
}

def Atir_MergeOp : Atir_Op<"Merge", Interfaces> {
  let summary = [{
    Accepts a list of input operands and returns a value of the operand 
    type and the index of the first encountered match.
  }];
  let arguments = (ins
    Variadic<AnyType>:$inputsAndControl
  );
  let results = (outs
    Atir_Tensor:$output,
    Atir_Tensor:$value_index
  );
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_BroadcastOp : Atir_Op<"broadcast",
    !listconcat([Pure], Interfaces)> {
  let summary = "Broadcast a tensor to a new shape by adding leading dimensions";
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$input,
    DenseI64ArrayAttr:$broadcastSizes
  );
  let results = (outs Atir_Tensor:$result);
  let assemblyFormat = [{
    $output `,` $input attr-dict `:` type($output) `,` type($input) `->` type($result)
  }];
}

def Atir_ReshapeOp : Atir_Op<"Reshape", !listconcat([Pure], Interfaces)> {
  let summary = "Reshape a tensor to a new shape without changing its underlying data.";
  let description = [{
    Produce a tensor with the same data as the input but reshaped to the specified shape.
    The total number of elements must remain unchanged. 
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$input,
    I64ArrayAttr:$targetShape
  );
  let results = (outs Atir_Tensor:$result);
  let assemblyFormat = [{
    $output `,` $input attr-dict `:` type($output) `,` type($input) `->` type($result)
  }];
}

def Atir_ParameterOp : Atir_Op<"Parameter", !listconcat([Pure], Interfaces)> {
  let summary = "Placeholder for input arguments of the computation graph";
  let description = [{
    This operation serves as a placeholder for external inputs into the computation graph.
    It is identified by a unique `parameterNumber` (unique globally within the computation graph) and specifies
    the type and shape of the tensor to be passed from the outside. This operation does not perform any computation;
    it only acts as an input anchor for the computation graph.
  }];
  let arguments = (ins I64Attr:$parameter_number);
  let results = (outs Atir_Tensor:$output);
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_CompareOp : Atir_Op<"Compare", !listconcat([Pure, ResultsBroadcastableShape], Interfaces)> {
  let summary = "Element-wise comparison operation.";
  let description = [{
    This operation compares two input tensors element-wise.
    The `comparisonDirection` attribute specifies the comparison relation.
    Supported values are: "EQ", "NE", "LT", "LE", "GT", "GE".
  }];
  let arguments = (ins
    Atir_Tensor:$lhs,
    Atir_Tensor:$rhs,
    StrAttr:$comparisonDirection
  );
  let results = (outs
    Atir_Tensor:$output
  );
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_DivideOp : Atir_Op<"Divide", !listconcat([Pure], Interfaces)> {
  let summary = "Element-wise division with broadcasting support";
  let description = [{
    Computes element-wise division of two tensors.
  }];
  let arguments = (ins 
    Atir_Tensor:$lhs, 
    Atir_Tensor:$rhs
  );
  let results = (outs Atir_Tensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_AndOp : Atir_Op<"And", !listconcat([Pure, ResultsBroadcastableShape], Interfaces)> {
  let summary = "Element-wise bitwise AND operation.";
  let description = [{
    This operation computes the bitwise AND of two input tensors
    element-wise. 
  }];
  let arguments = (ins
    Atir_Tensor:$lhs,
    Atir_Tensor:$rhs
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_LogisticOp : Atir_Op<"Logistic", [Pure, SameOperandsAndResultShape]> {
  let summary = "Element-wise logistic (sigmoid) function.";
  let description = [{
    This operation computes the logistic (sigmoid) function on
    each element of the input tensor: `1 / (1 + exp(-x))`.
  }];
  let arguments = (ins
    Atir_Tensor:$input
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_MaximumOp : Atir_Op<"Maximum", !listconcat([Pure, ResultsBroadcastableShape], Interfaces)> {
  let summary = "Element-wise maximum operation.";
  let description = [{
    This operation returns the element-wise maximum of two
    input tensors `lhs` and `rhs`. For each element, it compares the values
    and returns the larger one.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$lhs,
    Atir_Tensor:$rhs
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_ScatterOp : Atir_Op<"Scatter", !listconcat([Pure], Interfaces)> {
  let summary = "Scatter operation with configurable update semantics.";
  let description = [{
    This operation updates elements of the `operand` tensor at
    positions specified by `scatterIndices` using values from the `updates`
    tensor. The `updateComputation` region defines how to combine existing
    values with update values.
  }];
  let arguments = (ins
    Atir_Tensor:$operand,
    Atir_Tensor:$scatterIndices,
    Atir_Tensor:$updates,
    I64ArrayAttr:$updateWindowDims,
    I64ArrayAttr:$insertedWindowDims,
    I64ArrayAttr:$scatterDimsToOperandDims,
    DefaultValuedAttr<I64Attr, "0">:$indexVectorDim,
    DefaultValuedAttr<BoolAttr, "false">:$uniqueIndices,
    DefaultValuedAttr<BoolAttr, "false">:$indicesAreSorted
  );

  let results = (outs
    Atir_Tensor:$result
  );
  let regions = (region
    SizedRegion<1>:$updateComputation
  );
}

def Atir_AbsOp : Atir_Op<"Abs", [Pure, SameOperandsAndResultShape]> {
  let summary = "Element-wise absolute value operation.";
  let description = [{
    This operation computes the absolute value of each element in the input tensor. 
  }];
  let arguments = (ins
    Atir_Tensor:$input
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Atir_DotOp : Atir_Op<"Dot", !listconcat([Pure], Interfaces)> {
  let summary = "General dot product operation.";
  let description = [{
    This operation computes the dot product of two tensors `lhs` and `rhs`.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$lhs,
    Atir_Tensor:$rhs
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $lhs `,` $rhs attr-dict `:` type($output) `,` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def Atir_TransposeOp : Atir_Op<"Transpose", !listconcat([Pure], Interfaces)> {
  let summary = "Transpose a tensor by permuting its dimensions.";
  let description = [{
    This operation returns a tensor with dimensions permuted
    according to the `permutation` attribute.
  }];
  let arguments = (ins
    Atir_Tensor:$output,
    Atir_Tensor:$input,
    I64ArrayAttr:$permutation
  );
  let results = (outs
    Atir_Tensor:$result
  );
  let assemblyFormat = [{
    $output `,` $input attr-dict `:` type($output) `,` type($input) `->` type($result)
  }];
}

#endif  // ATIR_OPS_TD