# ONNX to StableHLO producer

This addon owns only the ONNX-to-StableHLO generation pipeline. Its stages are
kept separate by responsibility:

```text
ONNX model
  -> ONNX-MLIR import
  -> preprocessing passes (string boundary adaptation, ONNX canonicalization)
  -> ONNX-MLIR StableHLO lowering
  -> postprocessing passes (residual ONNX and shape legalization)
  -> StableHLO MLIR + boundary metadata
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
is outside the published StableHLO boundary.

`pipeline/` contains two executable programs because they do different jobs.
`annc-onnx2stablehlo` is the outer process driver: it validates files, invokes
the ONNX-MLIR tools in order, publishes the final StableHLO file, and writes
metadata. `annc-onnx-framework-opt` is the command-line entry point for the
addon passes: it registers those passes and calls `MlirOptMain` to read an MLIR
file and run the pass flags supplied by the driver. The pass code itself lives
under `passes/`. The upstream `onnx-mlir-opt` executable does not register
these ANNC passes, so the driver invokes this addon executable for preprocessing
and postprocessing; ONNX-MLIR remains responsible for its own StableHLO
conversion stage.

`annc-onnx-framework-opt` links ONNX-MLIR's `OMCompilerOptions` library. The
ONNX dialect libraries reference compiler-option definitions from that library,
including `disableMemRefPrefetch` and `getZArchNum`.
`passes/postprocessing/onnx_external_to_stablehlo.cpp` owns residual ONNX
lowering, including the patterns and the pass that applies them. The pass runs
once after the upstream StableHLO conversion.
This refactor ends at the framework-facing StableHLO artifact boundary; it does
not contain an ANNC reader or ATIR conversion.

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
producer is opt-in and does not alter the ONNX Runtime EP execution path.

Producer tests are in `tests/integration/onnx2stablehlo/`. They exercise the
pipeline boundary, preprocessing/postprocessing passes, and (when supplied)
real ORT-exported model files and portable StableHLO serialization.
