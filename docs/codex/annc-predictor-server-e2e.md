# ANNC Predictor Server E2E Verification

This local-only guide verifies the current ANNC checkout as an external
`LD_PRELOAD` plugin for `predictor_server`. The document is intentionally under
`docs/codex/`, which is ignored by Git.

## Prerequisites

- TensorFlow SDK: `/workspace/tensorflow/install`
- `predictor_server`: `/workspace/tensorflow/install/bin/predictor_server`
- Model: `/workspace/benchmark/wd_dcn/1`
- BRPC input data: `/workspace/benchmark/wd_dcn/wd_dcn.tsv`
- Session config: `/home/yyf/ANNC/tensorflow_addons/session_config.pbtxt`
- `/usr/local/bin/brpc_client`

## Build, Install And Preload Check

Run from `/home/yyf/ANNC`:

```bash
set -euo pipefail

export ANNC_TENSORFLOW_INCLUDE_DIR=/workspace/tensorflow/install/include
export ANNC_TENSORFLOW_CXX11_ABI=1
export ANNC_TENSORFLOW_PRELOAD=ON

./build.sh --clean \
  --build-type Release \
  --install-prefix /home/yyf/ANNC/install

TENSORFLOW_ROOT=/workspace/tensorflow/install \
ANNC_OPTIMIZER_SO=/home/yyf/ANNC/install/lib/libannc_optimizer.so \
  bash /workspace/tensorflow/tools/test_predictor_annc_plugin.sh
```

The top-level build includes `tensorflow_addons`, so this single build produces
both `libannc_optimizer.so` and `libannc_fused_op.so`; no separate
`tensorflow_addons` build is required. The final command must print
`external ANNC preload contract passed`.

## One-Second Request Smoke Test

Use the installed plugin and the freshly installed ANNC pipeline:

```bash
set -euo pipefail

OUT_DIR="${PWD}/logs/predictor-server-e2e-$(date +%Y%m%d_%H%M%S)"

env \
  OUT_DIR="${OUT_DIR}" \
  RUN_MODES=annc_plugin \
  ANNC_OPTIMIZER_SO=/home/yyf/ANNC/install/lib/libannc_optimizer.so \
  ANNC_PIPELINE_PATH=/home/yyf/ANNC/install/bin/annc-tf-pipeline \
  ANNC_SESSION_CONFIG=/home/yyf/ANNC/tensorflow_addons/session_config.pbtxt \
  ANNC_VERBOSE=1 \
  /workspace/tensorflow/tools/run_serving_benchmark.sh

rg -q 'ANNCOptimizer initialized' "${OUT_DIR}/annc_plugin/server.log"
rg -q 'Running ANNCOptimizer on graph' "${OUT_DIR}/annc_plugin/server.log"
rg -q 'Invoking annc-tf-pipeline graph rewrite' "${OUT_DIR}/annc_plugin/server.log"

echo "ANNC E2E verification passed: ${OUT_DIR}/annc_plugin/server.log"
```

The benchmark script stops the server after the request. A passing run reports
one successful request and all three optimizer log markers above.
