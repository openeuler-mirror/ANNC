//===----------------------------------------------------------------------===//
// Pimp Op definitions
//===----------------------------------------------------------------------===//
#ifndef PIMP_OPS_TD
#define PIMP_OPS_TD
include "Dialect/Pimp/PimpDialect.td"
include "Dialect/Pimp/Interfaces/ShapeInfer.td"
include "Dialect/Pimp/Interfaces/Interpret.td"
include "Dialect/Pimp/PimpEnums.td"

defvar Traits = [NoTerminator, NativeOpTrait<"Pimp_OpTrait">];
defvar Interfaces = [DeclareOpInterfaceMethods<ShapeInferIntfc>,
                     DeclareOpInterfaceMethods<InterpretIntfc>
                    ];

def MMLike : NativeOpTrait<"Pimp_MM">;
def DMALike : NativeOpTrait<"Pimp_DMA">;
def ACTLike : NativeOpTrait<"Pimp_ACTIVATION">;

class Pimp_Op<string mnemonic, list<Trait> traits = []> :
    Op<Pimp_Dialect, mnemonic, !listconcat(traits, Traits)> {
  let summary = "Astralax BaseOp";
  let results = (outs );
}

def Pimp_BufferOp : Pimp_Op<"buffer"> {
  let summary = "Allocate an uninitialized buffer for intermediate computation";
  let description = [{
    A placeholder for a temporary tensor buffer.
    The actual allocation is handled by the runtime or bufferization pass.
  }];

  let arguments = (ins);
  let results = (outs Pimp_Tensor:$output);
}

def Pimp_NoneOp : Pimp_Op<"None"> {
  let summary = "None op, return a NoneType Value";
  let description = [{ this op is used to define none type }];
  let results = (outs NoneType:$output);
  let assemblyFormat = [{ attr-dict }];
}

def Pimp_LoadOp : Pimp_Op<"Load", Interfaces> {
  let summary = "Load from a tensor";
  let description = [{Returns the element at the given location from within the tensor}];
  let arguments = (ins Pimp_Tensor:$input,
                    I32ArrayAttr:$axes,
                    I32ArrayAttr:$start,
                    I32ArrayAttr:$size,
                    BoolAttr:$copy
                    );
  let results = (outs Pimp_Tensor:$output);
}

def Pimp_CustomizeOp : Pimp_Op<"Customize"> {
  let summary = "customize op";
  let description = [{
    Customize op is defined for user defined operations.
    For example, 
      `FastAttention` op would be parsed into pimp.CustomizeOp.

    Since the custom op will not be optimized in most of general passes,
    convert it into the standard operations with `--graph-plugin` or 
    `--grouping-plugin` is highly recommend. 
  }];
  let arguments = (ins Variadic<Pimp_Tensor>:$inputs,
                       StrAttr:$opType);
  let results = (outs Variadic<Pimp_Tensor>:$output);
  let assemblyFormat = [{
    ( `(` $inputs^ `)` )?
    attr-dict `:` functional-type(operands, results)
  }];
}

def Pimp_ConstantOp : Pimp_Op<"constant", [DeclareOpInterfaceMethods<InterpretIntfc>]> {
  let summary = "declare or define a constant";
  let description = [{}];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let results = (outs Pimp_Tensor:$data);
  let assemblyFormat = [{
    ($sym_visibility^)? $sym_name 
    `->` type($data) attr-dict
  }];
}

def Pimp_VariableOp : Pimp_Op<"variable"> {
  let summary = "declare or define a variable";
  let description = [{}];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let results = (outs Pimp_Tensor:$output);
  let assemblyFormat = [{
    ($sym_visibility^)? $sym_name 
    `->` type($output) attr-dict
  }];
}

def Pimp_AddOp : Pimp_Op<"Add", Interfaces> {
  let summary = "add op";
  let arguments = (ins
      Variadic<Pimp_Tensor>:$inputs,
      DefaultValuedAttr<BoolAttr, "false">:$do_relu,
      DefaultValuedAttr<F32Attr, "-1.0">:$relu_limit,
      OptionalAttr<F32Attr>:$scalar
  );
  let results = (outs Pimp_Tensor:$output);
}

def Pimp_ConcatOp : Pimp_Op<"Concat", Interfaces> {
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
    ins Variadic<Pimp_Tensor>:$inputs,
    DefaultValuedAttr<I32Attr, "1">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F32Attr, "-1.0">:$relu_limit,
    DefaultValuedAttr<I32Attr, "0">:$round_mode,
    OptionalAttr<BoolAttr>:$is_view
  );

  let results = (outs Pimp_Tensor:$output);
}

def Pimp_MatMulOp : Pimp_Op<"MatMul", Interfaces> {
  let summary = "matmul op";
  let arguments = (
    ins Pimp_Tensor:$lhs,
    Pimp_Tensor:$rhs,
    Pimp_Tensor:$C,
    Optional<Pimp_Tensor>:$bias,
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
  let results = (outs Pimp_Tensor:$output);
}

def Pimp_ReluOp : Pimp_Op<"Relu", Interfaces> {
  let summary = "relu op";
  let arguments = (
      ins Pimp_Tensor:$input,
      DefaultValuedAttr<F32Attr, "-1.0">:$relu_limit
  );
  let results = (outs Pimp_Tensor:$output);
}

def Pimp_ForOp : Pimp_Op<"for"> {
  let summary = "for operation";
  let description = [{
    %0 = pimp.alloc()
    // concat
    pimp.for (i = 0; i < n; i+=s) {
      %in = extract() {[0, %i, 0 , 0] [?, s, ?, ?]}
      %1 = conv(%in)
      %2 = conv(%1)
      insert(%0, %2)
    }

    // reduce
    %0 = pimp.alloc()
    pimp.for (i = 0; i < n; i+=s; post_operation = ['reduce_sum'], iter_args = [%0]) {
      %in = extract() {[0, %i, 0 , 0] [?, s, ?, ?]}
      %1 = conv(%in)
      %2 = conv(%1)
      return %2
    }

    %0 = pimp.alloc()
    %1 = pimp.alloc()
    axt.for (i in [])
    pimp.for (i = 0; i < n; i+=s; post_operation = ['reduce_sum'], iter_args = [%0, %1], parallel) {
      %in = extract() {[0, %i, 0 , 0] [?, s, ?, ?]}
      %1 = conv(%in)
      %2 = conv(%in)
      return %1, %2
    }

    %start_in = arith.Constant{[0, 7]}
    %step_in = arith.Constant{[9, 8]}
    %start_out = arith.Constant([0, 4])
    %step_out = arith.Constant([4, 4])
    pimp.for %arg_start_in, %arg_step_in, %arg_start_out, %arg_step_out in zip(%start_in, %step_in, %start_out, %step_out) {
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
  let results = (outs Variadic<Pimp_Tensor>:$results);
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

def Pimp_IfOp : Pimp_Op<"if"> {
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

def Pimp_SwitchCaseOp : Pimp_Op<"switch", [SingleBlockImplicitTerminator<"pimp::ReturnOp">]> {
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

def Pimp_ParallelOp : Pimp_Op<"Parallel"> {
  let summary = "parallel group";
  let arguments = (ins Variadic<Pimp_Tensor>:$inputs);
  let results = (outs Variadic<Pimp_Tensor>:$outputs);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
    ( `(` $inputs^ `)` )?
    attr-dict `:` functional-type(operands, results)
    $region
  }];
}

def Pimp_ReturnOp : Pimp_Op<"return", [Terminator, ParentOneOf<["ForOp", "ParallelOp", "SwitchCaseOp"]>]> {
  let summary = "loop yield and termination operation";
  let description = [{}];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];
}

#endif  // PIMP_OPS_TD