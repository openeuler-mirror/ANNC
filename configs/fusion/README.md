# External Fusion Configs

`kp_embedding_patterns.json` extracts the KP embedding fusion operators from
PR 202 without importing its hard-coded first-level `OpFusion` matchers. Use it
with either entry point:

```shell
annc-opt input.mlir \
  --atir-config-fusion="config=configs/fusion/kp_embedding_patterns.json run-builtin-after=true"

annc-tf-pipeline \
  --input_graphdef input.pb \
  --output_graphdef output.pb \
  --fusion-config configs/fusion/kp_embedding_patterns.json
```

The normal `annc_execution_v2` rules outline function inputs and results.
Legacy kernels that receive ATIR output buffers use `abi: "mlir_ciface"` with
`buffer_outputs: true`; their declared output buffers are appended to the
outlined function arguments, preserving the existing FastCodegen and kernel
ABI.

The checked-in dynamic-stitch rule describes the N=2 form used by the focused
test. Backend implementations and registrations for N=2 through N=16 remain
available; model-specific configs can add the corresponding repeated
DynamicPartition/Gather branches without changing compiler code.

Scalar constant operands can be constrained directly in the graph DSL. Keep
the named operand in its physical position and omit it from `inputs`:

```json
"graph": ["y = Mul(x, scale=0.5)"],
"inputs": ["x"]
```

The matcher requires the corresponding operand to be an ATIR `ConstantOp`
whose scalar value equals the configured literal; a mismatch leaves the graph
unfused.
