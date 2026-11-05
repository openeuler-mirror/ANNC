//===----------------------------------------------------------------------===//
// ATIR V2 — complement dialect (only ops stablehlo cannot express)
//
// Admission rule (2026-11-02 decision, supersedes the A1/A2 layering; batch 5
// added the view domain):
//   Only capability domains stablehlo cannot express: string-specific /
//   resource / sparse / top_k (no primitive and no decomposition in this
//   pipeline) / constant_v2 (string constant carrier) / view domain
//   (get_dim/set_dim/encode_string/decode_string, carried across annc.*
//   boundaries). Ops stablehlo can express never enter this file — instances
//   fall back to the atir.unsupported partition semantics. The emit side is
//   guarded by StablehloSink's orphan defense: every emitted atir.* name must
//   exist in the dialect registry. Full decision and cost log:
//   schema-evolution.md (batches 4 and 5).
//===----------------------------------------------------------------------===//
#ifndef ATIR_OPS_V2_TD
#define ATIR_OPS_V2_TD

def Atir_ResourceIdentityV2Op : Atir_Op<"resource_identity_v2"> {
  let summary = "forwards an opaque resource handle";
  let arguments = (ins Atir_Resource:$input);
  let results = (outs Atir_Resource:$result);
}

def Atir_ResourceVariableV2Op : Atir_Op<"resource_variable_v2"> {
  let summary = "declares a runtime resource handle";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility
  );
  let results = (outs Atir_Resource:$handle);
}

def Atir_ConstantV2Op : Atir_Op<"constant_v2"> {
  let summary = "declare or define a constant";
  let description = [{}];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility
  );
  let results = (outs AnyTensor:$data);
}

def Atir_VariableV2Op : Atir_Op<"variable_v2"> {
  let summary = "declare or define a variable";
  let description = [{}];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility
  );
  let results = (outs AnyTensor:$output);
}

def Atir_StringToHashBucketFastV2Op : Atir_Op<"string_to_hash_bucket_fast_v2",
    [SameOperandsAndResultShape]> {
  let summary = "Map strings to hash bucket indices.";
  let description = [{
    Maps each string in `input` to an integer hash modulo `numBuckets`.
    Hash function is deterministic within a run but not guaranteed across versions.
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$numBuckets
  );
  let results = (outs
    AnyTensor:$result
  );
}

def Atir_StringToNumberV2Op : Atir_Op<"string_to_number_v2",
    [SameOperandsAndResultShape]> {
  // dtype: string-only (D)
  let summary = "Convert each string in the input tensor to a numeric value.";
  let description = [{
    Converts each string element of `input` to a numeric value. The output
    element type is fixed by the frontend (TF `out_type` attribute); the op
    itself only requires the input to be a string tensor (encoding="string").
  }];
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}

def Atir_StaticRegexReplaceV2Op : Atir_Op<"static_regex_replace_v2",
    [SameOperandsAndResultShape]> {
  // dtype: string-only (D)
  let summary = "Replaces matches of a regular expression in each string.";
  let description = [{
    Replaces matches of `pattern` in each string element of `input` with
    `rewrite`. `pattern` follows the RE2-compatible syntax used by TF. When
    `replaceGlobal` is set, every match is replaced; otherwise only the first
    match is replaced. The result carries string elements.
  }];
  let arguments = (ins
    AnyTensor:$input,
    StrAttr:$pattern,
    StrAttr:$rewrite,
    DefaultValuedAttr<BoolAttr, "true">:$replaceGlobal
  );
  let results = (outs AnyTensor:$result);
}

def Atir_StringSplitV2Op : Atir_Op<"string_split_v2"> {
  // dtype: string-only (D)
  let summary = "Split a string tensor by a delimiter into a SparseTensor.";
  let description = [{
    Splits each string element of `input` by the scalar string `delimiter`
    producing three outputs: `indices` [nnz, rank+1] (int64), `values`
    (the split substrings, string elements) and `denseShape` [rank+1] (int64
    where the trailing dim is the max substring count). When `skipEmpty` is
    set, empty substrings are dropped.
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$delimiter,
    DefaultValuedAttr<BoolAttr, "false">:$skipEmpty
  );
  let results = (outs
    AnyTensor:$indices,
    AnyTensor:$values,
    AnyTensor:$denseShape
  );
}

def Atir_SparseTensorDenseMatMulV2Op : Atir_Op<"sparse_tensor_dense_mat_mul_v2"> {
  // dtype: sparse (E)
  let summary = "Multiply a sparse matrix by a dense matrix.";
  let description = [{
    Computes `A * B` where `A` is a sparse matrix given by `indices` [nnz, 2]
    (int64), `values` [nnz] and `denseShape` [M, N], and `B` is the dense
    [N, K] matrix. The result is [M, K]. `adjointA`/`adjointB` conjugate-
    transpose (a transpose for real values) the respective operand first.
  }];
  let arguments = (ins
    AnyTensor:$indices,
    AnyTensor:$values,
    AnyTensor:$denseShape,
    AnyTensor:$dense,
    DefaultValuedAttr<BoolAttr, "false">:$adjointA,
    DefaultValuedAttr<BoolAttr, "false">:$adjointB
  );
  let results = (outs AnyTensor:$result);
}

def Atir_SparseReshapeV2Op : Atir_Op<"sparse_reshape_v2"> {
  // dtype: sparse (E)
  let summary = "Reshape a SparseTensor to a new dense shape.";
  let description = [{
    This operation reshapes a sparse tensor represented by indices and values
    to a new dense shape. The non-zero elements remain the same, only their
    positions change according to the new shape.
  }];
  let arguments = (ins
    AnyTensor:$indices,
    AnyTensor:$inputShape,
    AnyTensor:$newShape
  );
  let results = (outs
    AnyTensor:$outputIndices,
    AnyTensor:$outputShape
  );
}

def Atir_SparseFillEmptyRowsV2Op : Atir_Op<"sparse_fill_empty_rows_v2"> {
  // dtype: sparse (E)
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
  let arguments = (ins AnyTensor:$indices,      // [N, rank] int64
    AnyTensor:$values,                          // [N] any type T
    AnyTensor:$denseShape,                      // [rank] int64
    AnyTensor:$defaultValue);                   // scalar of type T
  let results = (outs
    AnyTensor:$outputIndices,                   // [N_full, rank] int64
    AnyTensor:$outputValues,                    // [N_full] same type as values
    AnyTensor:$emptyRowIndicator,               // [dense_shape[0]] bool
    AnyTensor:$reverseIndexMap                  // [N] int64
  );
}

def Atir_VarHandleV2Op : Atir_Op<"var_handle_op_v2", [NoMemoryEffect]> {
  // dtype: capability (C)
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
  // Results are pending resource-domain landing: the handle will be declared
  // as `Atir_Resource` once the read/assign consumers are migrated. Until
  // then, emissions to this op fall back to the unsupported carrier (the
  // local verify rejects the arity mismatch).
}

def Atir_ResourceGatherV2Op : Atir_Op<"resource_gather_v2"> {
  // dtype: capability (C)
  let summary = "Gather elements from resource handle by indices";
  let description = [{
    Gathers data from a resource according to given indices.
    Supports batched gathering with multiple index dimensions.
  }];
  let arguments = (ins
    AnyType:$resource,
    Variadic<AnyType>:$indices,
    DefaultValuedAttr<BoolAttr, "false">:$validateIndices,
    DefaultValuedAttr<I64Attr, "0">:$batchDims
  );
  let results = (outs AnyTensor:$result);
}

def Atir_SparseSegmentSumV2Op : Atir_Op<"sparse_segment_sum_v2"> {
  // dtype: sparse (E)
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
    AnyTensor: $input,
    AnyTensor: $indices,
    AnyTensor: $segmentIds,
    AnyTensor:$numSegments
  );
  let results = (outs
    AnyTensor: $result
  );
}

def Atir_SparseSegmentMeanV2Op : Atir_Op<"sparse_segment_mean_v2"> {
  // dtype: sparse (E)
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
    AnyTensor:$data,
    AnyTensor:$indices,
    AnyTensor:$segmentIds,
    AnyTensor:$numSegments
  );
  let results = (outs
    AnyTensor:$result
  );
}

def Atir_SparseToDenseV2Op : Atir_Op<"sparse_to_dense_v2"> {
  // dtype: sparse (E)
  let summary = "Converts sparse representations into dense tensors.";
  let description = [{
    Converts sparse representation (sparseIndice + sparseValues) to a dense tensor with specified shape.
    The output tensor is initialized with defaultValue, then filled with values at the positions specified by indices.
    The index should be sorted in lexicographical order, and it should not contain any duplicate items.
    If `validateIndices` is true, these properties will be checked during the execution process.
  }];
  let arguments = (ins
    AnyTensor:$sparseIndice,
    AnyTensor:$outputShape,
    AnyTensor:$sparseValue,
    AnyTensor:$defaultValue,
    DefaultValuedOptionalAttr<BoolAttr, "true">:$validateIndices
  );
  let results = (outs
    AnyTensor:$result
  );
}

def Atir_TopKV2Op : Atir_Op<"top_k_v2"> {
  // dtype: pipeline-complement (G)
  let summary = "Find the values and indices of the `k` largest elements along the last dimension.";
  let description = [{
    Find the values and indices of the `k` largest elements along the last dimension.
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$k,
    DefaultValuedOptionalAttr<BoolAttr, "true">:$sorted
  );
  let results = (outs
    AnyTensor:$values,
    AnyTensor:$indices
  );
}

//===----------------------------------------------------------------------===//
// Frontend boundary views (annc.* ABI carriers; schema-evolution batch 5).
//
// The four frontend wrap boundaries (`annc.getDim` / `annc.setDim` /
// `annc.encodeString` / `annc.decodeString`, carried on tf.residue
// `call_target_name`) were previously emitted as atir.unsupported
// (reason=unknown_op), losing their semantics to string matching downstream.
// They are view/encoding carriers by construction — stablehlo cannot express
// them — so they enter this complement dialect as first-class ops. Their
// unranked/dynamic payloads are legal (that is what the views transport),
// which exempts them from the sink's unrank_blocked guard.
//===----------------------------------------------------------------------===//

def Atir_GetDimV2Op : Atir_Op<"get_dim_v2"> {
  // dtype: view (B)
  let summary = "one-dimensional sequence view of any tensor";
  let description = [{
    Reveals the row-major element sequence of `input` (ranked-dynamic or
    unranked) as a flat 1-D tensor `flat`, plus the runtime shape of `input`
    as an i64 vector `shape`. Zero-copy view: the element order and count are
    preserved, no dimension structure is assumed. Shape-agnostic ops execute
    on `flat` in the sequence domain; the second result feeds shape-consuming
    consumers or the `set_dim_v2` restore.
  }];
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$flat, AnyTensor:$shape);
}

def Atir_SetDimV2Op : Atir_Op<"set_dim_v2"> {
  // dtype: view (B)
  let summary = "restore a tensor from its one-dimensional sequence view";
  let description = [{
    Restores `input` (a flat 1-D sequence view, e.g. the `flat` result of
    `get_dim_v2`) back to a tensor. With `shape` (an integer vector) the
    result is ranked to that shape — this is the Reshape short-circuit:
    reshaping never reorders the row-major element sequence, so
    `set_dim_v2(get_dim_v2(x), shape)` is exactly `reshape(x, shape)` without
    any kernel work. Without `shape` the result is unranked and the runtime
    context supplies the layout (the legacy single-operand annc.setDim ABI).
  }];
  let arguments = (ins AnyTensor:$input, Optional<AnyTensor>:$shape);
  let results = (outs AnyTensor:$result);
}

def Atir_EncodeStringV2Op : Atir_Op<"encode_string_v2"> {
  // dtype: string-only (D)
  let summary = "encode string elements into fixed-width self-describing i8 rows";
  let description = [{
    Encodes each string element of `input` into one fixed-width byte row
    `[i32 length prefix][payload][zero padding]`; the result is an i8 tensor
    with one trailing row-width dimension appended to the input shape. Rows
    are opaque to the numeric domain: row-transport ops execute on the i8
    representation while the official legalize handles them natively.
  }];
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$rows);
}

def Atir_DecodeStringV2Op : Atir_Op<"decode_string_v2"> {
  // dtype: string-only (D)
  let summary = "decode fixed-width i8 rows back into string elements";
  let description = [{
    Inverse of `encode_string_v2`: reads each row's i32 length prefix,
    extracts the payload and produces string elements. The row-width
    dimension is consumed; the remaining shape matches the encoded tensor's
    logical shape.
  }];
  let arguments = (ins AnyTensor:$rows);
  let results = (outs AnyTensor:$result);
}

#endif  // ATIR_OPS_V2_TD
