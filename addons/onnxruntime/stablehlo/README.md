# ONNX to StableHLO producer

This addon owns only the ONNX-to-StableHLO generation pipeline. Its stages are
kept separate by responsibility:

```text
ONNX model
  -> framework-opt: import ONNX once and retain the original graph index
  -> ONNX IR optimization (decomposition, shape inference, canonicalization)
  -> preprocessing passes (string boundary adaptation, Slice, dynamic Tile)
  -> ONNX-MLIR StableHLO lowering
  -> postprocessing passes (residual ONNX and shape legalization)
  -> StableHLO / supported ATIR fallbacks + boundary metadata
```

`passes/preprocessing/` contains transformations that prepare imported ONNX
for StableHLO lowering. The current pipeline runs these three passes, in order:

1. `onnx-string-to-framework-custom-call`: changes string `Constant`/`Expand`
   into the generic `annc_framework.custom_call` placeholder with lossless
   metadata.
2. `onnx-slice-preprocess`: normalizes Slice index operands and default axes/
   steps for upstream lowering.
3. `onnx-dynamic-tile-preprocess`: replaces dynamic `onnx.Tile` with an
   internal marker understood by residual lowering.

The upstream ONNX-MLIR optimizer then runs `--convert-onnx-to-stablehlo`.
After that, the addon pass executable runs
`framework-custom-call-to-stablehlo`, `onnx-external-to-stablehlo`,
`framework-shape-to-stablehlo`, canonicalization, and
`verify-framework-stablehlo`. The last stage rejects any operation or type that
is outside the published StableHLO / supported ATIR fallback boundary.

`pipeline/` contains two executable programs because they do different jobs.
`annc-onnx2stablehlo` validates files, imports the original model once through
`annc-onnx-framework-opt --annc-import-onnx`, and passes the indexed ONNX IR to
`onnx-mlir-opt`. It then runs preprocessing, StableHLO lowering, postprocessing
and source finalization before publishing the IR and required `--metadata`
diagnostics. The input model, IR output and metadata must refer to different
files. Path checks identify `..`, directory symlinks and existing hard links;
rejected destinations leave existing files unchanged. Both output parent
directories must already exist.

`annc-onnx-framework-opt` registers the addon import and passes. Outside import
mode, it calls `MlirOptMain` to read MLIR and run the supplied pass flags. The
upstream `onnx-mlir-opt` executable does not register these ANNC passes, so the
driver invokes the addon tool for preprocessing and postprocessing. ONNX-MLIR
owns its StableHLO conversion stage.

`annc-onnx-framework-opt` links ONNX-MLIR's `OMCompilerOptions` library. The
ONNX dialect libraries reference compiler-option definitions from that library,
including `disableMemRefPrefetch` and `getZArchNum`.
`passes/postprocessing/onnx_external_to_stablehlo.cpp` owns residual ONNX
lowering, including the patterns and the pass that applies them. The pass runs
once after the upstream StableHLO conversion.
Residual `Mod` computes a joint ONNX broadcast shape and broadcasts either
input as needed. It reuses shape legalization and preserves integer modulo
and floating remainder sign rules. Equal dynamic tensor types still require
broadcasting because their concrete shapes can differ.

The supported framework-owned `com.microsoft.FusedTensordotMatMul` remains an
`atir.opaque` with its original attributes and identities. Use the paired
`annc-onnx-framework-opt` to read or round-trip mixed IR; portable StableHLO
serialization applies to pure HLO. The addon produces the converted IR with
source attributes. For fusion, backend compilation and ORT execution, see the
[ONNX pipeline workflow](../../../docs/guides/cli-workflows.md#onnx-的-stablehlo-前端).

ONNX-MLIR v0.5.0.2 is pinned as the `third-party/onnx-mlir` Git submodule at
`128f8b716fdbed4ff05a5ba493ac48fcac761b40`. Its nested StableHLO submodule is
pinned with it. The submodule provides the source; CMake must still build the
ONNX-MLIR tools. Initialize the sources with:

```sh
git submodule update --init --recursive third-party/onnx-mlir
```

The build script constructs the complete LLVM/MLIR components required by the
pinned ONNX-MLIR revision, followed by ONNX-MLIR and its paired StableHLO
tools. It does not require a prebuilt LLVM/MLIR installation. It does require
Git, CMake, Ninja, a C/C++ compiler, Python development headers, and the system
protobuf compiler/library.

```sh
addons/onnxruntime/stablehlo/build_onnx_mlir.sh --jobs 16

cmake -S . -B build \
  -DLLVM_DIR="$PWD/build/third-party/llvm-build/lib/cmake/llvm" \
  -DMLIR_DIR="$PWD/build/third-party/llvm-build/lib/cmake/mlir" \
  -DANNC_ENABLE_ONNXRUNTIME_STABLEHLO=ON \
  -DANNC_ONNX_MLIR_BUILD_DIR="$PWD/build/third-party/onnx-mlir-build"
cmake --build build --target annc_onnx_stablehlo_addon -j16
```

The script checks out the llvm-project revision paired with the ONNX-MLIR
submodule (`b270525f730be6e7196667925f5a9bfa153262e9`) and caches its source
and build under `build/third-party/`. `--llvm-source` can point to an existing
checkout of that exact revision when network access is unavailable. The
producer can also be used independently to inspect the converted IR.

Producer tests are in `tests/integration/onnx2stablehlo/`. They exercise the
pipeline boundary, preprocessing/postprocessing passes, and (when supplied)
real ORT-exported model files and portable StableHLO serialization.
