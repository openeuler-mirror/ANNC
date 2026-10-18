#!/usr/bin/env python3
"""TensorFlow predictor + ANNC addon end-to-end smoke test.

This is a manual/CI-environment test: it needs an external predictor_server
built with the ANNC host contract and a SavedModel/TSV pair.  It verifies the
layered contract that a plain `success` from brpc_client does not prove:

  1. ANNCOptimizer is loaded and the graph rewrite ran;
  2. no JIT kernel failed to compile and no model run failed;
  3. at least one JIT kernel .so was produced in ANNC_WORK_DIR.

Required environment:
  ANNC_PREFIX    install prefix containing bin/annc-tf-pipeline, lib/libannc_optimizer.so
  ANNC_SRC       ANNC source tree (session config + GEMM config)
  TF_PREDICTOR   predictor_server executable
  TF_MODEL       SavedModel directory
  TF_DATA        benchmark TSV (header + id<TAB>model_input)
  BRPC_CLIENT    brpc_client executable

Optional: SYSTEM_LIB_DIR (appended to LD_LIBRARY_PATH), PORT, ANNC_WORK_DIR.
"""
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def env(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing required environment variable {name}")
    return value


prefix = Path(env("ANNC_PREFIX"))
src = Path(env("ANNC_SRC"))
predictor = env("TF_PREDICTOR")
model = env("TF_MODEL")
data = env("TF_DATA")
client = env("BRPC_CLIENT")
port = os.environ.get("PORT", "29991")
work_dir = Path(os.environ.get(
    "ANNC_WORK_DIR", tempfile.mkdtemp(prefix="annc_tf_predictor_e2e_")))
work_dir.mkdir(parents=True, exist_ok=True)
log_path = work_dir / "predictor.log"

run_env = os.environ.copy()
run_env.pop("LD_PRELOAD", None)
run_env.update({
    "PATH": f"{prefix}/bin:" + run_env.get("PATH", ""),
    "LD_LIBRARY_PATH": ":".join(
        p for p in (f"{prefix}/lib", os.environ.get("SYSTEM_LIB_DIR", "")) if p),
    "LD_PRELOAD": f"{prefix}/lib/libannc_optimizer.so",
    "ANNC_ENABLE": "1",
    "ANNC_VERBOSE": "1",
    "ANNC_PIPELINE_PATH": f"{prefix}/bin/annc-tf-pipeline",
    "ANNC_FUSED_OP_PATH": f"{prefix}/lib/libannc_fused_op.so",
    "ANNC_WORK_DIR": str(work_dir),
    "ANNC_GEMM_CONFIG": f"{src}/benchmark/annc-bench/configs/gemm-hip09-multi.json",
    "ANNC_TIMEOUT": "1800",
    "TF_NUM_INTEROP_THREADS": "16",
    "TF_NUM_INTRAOP_THREADS": "16",
    "TF_ENABLE_ONEDNN_OPTS": "0",
})

cmd = [
    predictor,
    "--enable_kdnn=false",
    f"--model_path={model}",
    "--thread_num=16",
    "--batch_size=100",
    f"--port={port}",
    f"--session_config_pbtxt={src}/tensorflow_addons/session_config.pbtxt",
]
with open(log_path, "w") as log:
    proc = subprocess.Popen(cmd, env=run_env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.time() + 180
        while time.time() < deadline:
            if proc.poll() is not None:
                raise SystemExit(f"predictor exited early; see {log_path}")
            if f"is serving on port={port}" in log_path.read_text(errors="replace"):
                break
            time.sleep(1)
        else:
            raise SystemExit(f"predictor did not become ready; see {log_path}")
        subprocess.run([
            client, f"--server=127.0.0.1:{port}", f"--input_data_path={data}",
            "--thread_num=1", "--warmup_duration_s=0", "--test_duration_s=1",
            "--max_qps=1", "--max_inflight=1",
            "--request_min_batch_size=1", "--request_max_batch_size=1",
            "--timeout_ms=60000",
        ], check=True)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=20)

text = log_path.read_text(errors="replace")
checks = {
    "ANNCOptimizer initialized": "ANNCOptimizer initialized" in text,
    "graph rewrite completed": "graph rewrite completed successfully" in text,
    "no compile_failed": "compile_failed" not in text,
    "no Running model failed": "Running model failed" not in text,
    "JIT kernel .so produced": any(work_dir.rglob("kernel.so")),
}
for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
if not all(checks.values()):
    print(f"log: {log_path}")
    sys.exit(1)
print(f"PASS: TF predictor ANNC e2e ({work_dir})")
