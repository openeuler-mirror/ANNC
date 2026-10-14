#!/usr/bin/env python3
"""Exercise the annc_execution_v2 EP path with a hand-written context kernel.

The fixture stands in for `annc-onnx-pipeline` (like ep_aot_fixture.py does for
the preallocated-output ABI), so the real graph export, site validation, cache
publication, ORT fusion and Compute run unchanged; only the kernel is a local
stand-in. The kernel drives every piece of the execution context:

  * it allocates its output through allocate_output with dims it reads from the
    runtime descriptors (the leading dims are dynamic in the model, so only the
    kernel can size the buffer),
  * it allocates a zero-filled temporary through allocate_temp and computes
    into it before copying to the output,
  * it runs the rows through the thread pool the context carries and records
    the distinct worker threads it observed,
  * on demand it reports an error through report_error or requests an
    out-of-range output slot, to pin the failure paths.

Modes (ANNC_EP_FIXTURE_KERNEL_MODE):
  normal    matmul into the context-allocated output (the default)
  error     report_error + non-zero return
  bad-slot  allocate output slot 1 (out of range)
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import onnxruntime.capi.onnxruntime_pybind11_state as rtpy


KERNEL = r"""
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "Kernel/ExecutionContext.h"
#include "Support/ThreadPool/ThreadPool.h"

// The EP calls the kernel with one ranked-memref descriptor per site argument.
struct MemRef3D {
  float *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
struct MemRef2D {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};

namespace {

void AppendMarker(const char *line) {
  const char *path = std::getenv("ANNC_EP_FIXTURE_MARKER");
  if (path == nullptr) return;
  FILE *out = std::fopen(path, "a");
  if (out != nullptr) {
    std::fputs(line, out);
    std::fputc('\n', out);
    std::fclose(out);
  }
}

const char *KernelMode() {
  const char *mode = std::getenv("ANNC_EP_FIXTURE_KERNEL_MODE");
  return mode != nullptr ? mode : "normal";
}

}  // namespace

extern "C" int32_t _mlir_ciface_fixture_context_matmul(
    const AnncExecutionContext *context, void *input_arg, void *weight_arg) {
  if (context == nullptr) return 1;
  MemRef3D *x = static_cast<MemRef3D *>(input_arg);
  MemRef2D *w = static_cast<MemRef2D *>(weight_arg);
  const int64_t d0 = x->sizes[0];
  const int64_t d1 = x->sizes[1];
  const int64_t k = x->sizes[2];
  const int64_t n = w->sizes[1];
  const int64_t m = d0 * d1;

  if (std::strcmp(KernelMode(), "error") == 0) {
    context->report_error(context->handle, ANNC_STATUS_INTERNAL,
                          "fixture: deliberate kernel error");
    AppendMarker("error deliberate");
    return 7;
  }

  // The output's leading dims are dynamic in the model, so the kernel has to
  // read the runtime descriptors before it can size the buffer.
  const uint32_t slot =
      std::strcmp(KernelMode(), "bad-slot") == 0 ? 1u : 0u;
  int64_t out_dims[3] = {d0, d1, n};
  AnncTensorDesc out_desc{sizeof(AnncTensorDesc), ANNC_ELEMENT_TYPE_F32, 3,
                          ANNC_ALLOCATION_NONE, out_dims, nullptr};
  AnncStatusCode status = context->allocate_output(context->handle, slot,
                                                   &out_desc);
  if (status != ANNC_STATUS_OK) return 3;

  int64_t scratch_dims[2] = {m, n};
  AnncTensorDesc temp_desc{sizeof(AnncTensorDesc), ANNC_ELEMENT_TYPE_F32, 2,
                           ANNC_ALLOCATION_ZERO, scratch_dims, nullptr};
  status = context->allocate_temp(context->handle, &temp_desc);
  if (status != ANNC_STATUS_OK) return 4;
  float *scratch = static_cast<float *>(temp_desc.data);

  auto *pool = reinterpret_cast<annc::threadpool::AnncThreadPool *>(
      context->thread_pool);
  if (pool == nullptr) {
    context->report_error(context->handle, ANNC_STATUS_INTERNAL,
                          "fixture: context carries no thread pool");
    return 5;
  }

  std::mutex mutex;
  std::vector<std::thread::id> workers;
  int nested_calls = 0;
  int nested_in_region = -1;
  std::vector<std::thread::id> nested_ids;
  pool->parallel_for(
      m, annc::threadpool::ParallelForOptions{},
      [&](int64_t begin, int64_t end) {
        {
          std::lock_guard<std::mutex> lock(mutex);
          const std::thread::id id = std::this_thread::get_id();
          if (std::find(workers.begin(), workers.end(), id) == workers.end()) {
            workers.push_back(id);
          }
        }
        if (std::strcmp(KernelMode(), "nested") == 0) {
          // Emulate the gate ANNC's own parallel_for applies (Parallel.cpp):
          // inside a region it runs the whole range serially. The fixture
          // does not link the ANNC runtime, so the gate is spelled out here;
          // what it demonstrates is that in_parallel_region() reports true
          // while this callback runs. The nested range is deliberately large
          // so that a missing gate would visibly fan out, and the distinct
          // threads it ran on are recorded.
          const bool in_region = pool->in_parallel_region();
          {
            std::lock_guard<std::mutex> lock(mutex);
            if (nested_in_region < 0) nested_in_region = in_region ? 1 : 0;
            ++nested_calls;
          }
          auto nested_body = [&](int64_t, int64_t) {
            std::lock_guard<std::mutex> lock(mutex);
            if (std::find(nested_ids.begin(), nested_ids.end(),
                          std::this_thread::get_id()) == nested_ids.end()) {
              nested_ids.push_back(std::this_thread::get_id());
            }
          };
          if (in_region) {
            nested_body(0, 4096);
          } else {
            pool->parallel_for(4096, annc::threadpool::ParallelForOptions{},
                               nested_body);
          }
        }
        for (int64_t row = begin; row < end; ++row) {
          const int64_t i = d1 > 0 ? row / d1 : 0;
          const int64_t j = d1 > 0 ? row % d1 : 0;
          const float *xr = x->aligned + x->offset + i * x->strides[0] +
                            j * x->strides[1];
          float *sr = scratch + row * n;
          for (int64_t col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int64_t t = 0; t < k; ++t) {
              sum += xr[t * x->strides[2]] *
                     w->aligned[w->offset + t * w->strides[0] +
                                col * w->strides[1]];
            }
            sr[col] = sum;
          }
        }
      });

  float *out = static_cast<float *>(out_desc.data);
  if (m > 0 && n > 0) {
    std::memcpy(out, scratch, static_cast<size_t>(m) * n * sizeof(float));
  }

  char line[160];
  std::snprintf(line, sizeof(line),
                "compute dims=%lld,%lld,%lld threads=%zu nested=%d,%d,%zu",
                static_cast<long long>(d0), static_cast<long long>(d1),
                static_cast<long long>(n), workers.size(), nested_calls,
                nested_in_region, nested_ids.size());
  AppendMarker(line);
  return 0;
}
"""


def run_pipeline(args):
    if args.toolchain_info:
        print("annc-ep-context-fixture-v1")
        return
    model = onnx.load(args.model)
    onnx.checker.check_model(model)
    nodes = list(model.graph.node)
    if len(nodes) != 1 or nodes[0].op_type != "MatMul":
        raise ValueError("fixture requires exactly one exported MatMul")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    kernel_name = "fixture_context_matmul"
    source = output / "kernel.cpp"
    source.write_text(KERNEL)
    include_dir = Path(__file__).resolve().parents[3] / "annc" / "include"
    subprocess.run(["c++", "-std=c++17", "-shared", "-fPIC", "-O2",
                    "-I", str(include_dir), "-o",
                    str(output / "libannc_kernels.so"), str(source)],
                   check=True)
    columns = model.graph.initializer[0].dims[1]
    site = {
        "pattern": "tensordot_matmul", "execution_mode": "aot",
        "abi": "annc_execution_v2",
        "kernel_name": kernel_name, "member_node_names": [nodes[0].name],
        "kernel_arg_order": [],
        "args": [
            {"tf_name": nodes[0].input[0], "role": "dynamic", "dtype": "f32",
             "rank": 3, "shape": [-1, -1, -1]},
            {"tf_name": nodes[0].input[1], "role": "fixed", "dtype": "f32",
             "rank": 2, "shape": [4, columns]},
        ],
        "outputs": [{"tf_name": nodes[0].output[0], "role": "output",
                     "dtype": "f32", "rank": 3, "shape": [-1, -1, -1]}],
    }
    (output / "sites.json").write_text(json.dumps({
        "schema": "annc-ep-site-report/1", "contract_version": "1.0.0",
        "sites": [site],
    }))


def make_model(path, batch):
    weight = np.arange(20, dtype=np.float32).reshape(4, 5) / 7
    # The batch dimension is dynamic on purpose: the context-allocated output
    # size must follow the runtime input, so the test can vary it per run.
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="n0")],
        "fixture",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT,
                                       [None, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT,
                                       [None, 3, 5])],
        [numpy_helper.from_array(weight, "w")])
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    proto.ir_version = 7
    onnx.checker.check_model(proto)
    onnx.save(proto, path)


def session(model, providers, threads=1):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model), options, providers=providers)


def run_test(args):
    work = Path(args.work_dir or tempfile.mkdtemp(prefix="annc-ep-context-"))
    work.mkdir(parents=True, exist_ok=True)
    batch = 128 if args.mode == "threads" else 2
    model = work / "matmul.onnx"
    make_model(model, batch)
    x = np.arange(batch * 12, dtype=np.float32).reshape(batch, 3, 4) / 11

    runtime = Path(rtpy.__file__).parent / "libonnxruntime.so.1.26.0"
    ctypes.CDLL(str(runtime), mode=ctypes.RTLD_GLOBAL)
    rtpy.register_execution_provider_library("AnncExecutionProvider",
                                             args.ep_library)
    os.environ.update({
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "ANNC_EP_PIPELINE": str(Path(__file__).resolve()),
        "ANNC_EP_CACHE_DIR": str(work / "cache"),
        "ANNC_EP_FIXTURE_MARKER": str(work / "compute.log"),
        "ANNC_EP_FIXTURE_KERNEL_MODE": args.mode,
        "ANNC_EP_DEBUG": "1",
    })
    expected = session(model, ["CPUExecutionProvider"]).run(None, {"x": x})[0]
    marker = work / "compute.log"

    if args.mode == "error":
        try:
            session(model, ["AnncExecutionProvider", "CPUExecutionProvider"]).run(
                None, {"x": x})
            raise AssertionError("the reported error did not fail the run")
        except Exception as error:  # noqa: BLE001 - the message is the contract
            assert "deliberate kernel error" in str(error), error
        assert marker.read_text().splitlines() == ["error deliberate"], marker
        # The host survives and falls back to the CPU EP for the same model.
        actual = session(model, ["CPUExecutionProvider"]).run(None, {"x": x})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-7)
        print("Kernel error: surfaced as an OrtStatus; host intact")
        return

    if args.mode == "bad-slot":
        try:
            session(model, ["AnncExecutionProvider", "CPUExecutionProvider"]).run(
                None, {"x": x})
            raise AssertionError("the out-of-range slot did not fail the run")
        except Exception as error:  # noqa: BLE001
            assert "output slot 1 is out of range" in str(error), error
        print("Bad output slot: rejected by the context callbacks")
        return

    if args.mode == "threads":
        # intra-op=1: the pool ORT exposes has no worker, the region runs
        # serially on this thread.
        session(model, ["AnncExecutionProvider", "CPUExecutionProvider"],
                threads=1).run(None, {"x": x})
        single = marker.read_text().splitlines()
        assert single[0].startswith(f"compute dims={batch},3,5 threads=1"), single
        # intra-op=4: the same kernel now runs inside ORT's pool. The pool
        # creates its workers lazily and the shards are tiny, so the first run
        # may still finish on the calling thread; on a loaded machine
        # (concurrent builds or test suites saturating every core) several
        # sessions can do that. Give the observation room before calling the
        # pool ineffective -- this is a best-effort signal, and stopping at
        # the first session that fans out keeps the passing path fast.
        observed = 1
        for _ in range(8):
            session(model, ["AnncExecutionProvider", "CPUExecutionProvider"],
                    threads=4).run(None, {"x": x})
            lines = marker.read_text().splitlines()
            observed = max(observed,
                           int(lines[-1].split("threads=", 1)[1].split()[0]))
            if observed >= 2:
                break
        assert observed >= 2, lines
        print(f"Thread pool: 1 thread serially, {observed} workers with intra-op=4")
        return

    if args.mode == "nested":
        session(model, ["AnncExecutionProvider", "CPUExecutionProvider"],
                threads=4).run(None, {"x": x})
        line = marker.read_text().splitlines()[-1]
        nested_calls, nested_in_region, nested_workers = (
            int(part) for part in line.split("nested=", 1)[1].split(","))
        # One nested region per row; each reports in_parallel_region() == true,
        # and the gated nested range of 4096 items ran on a single thread --
        # a missing gate would have fanned it out like the outer region.
        assert nested_calls == batch * 3, line
        assert nested_in_region == 1, line
        assert nested_workers == 1, line
        print("Nested parallel region: reported and run without fanning out")
        return

    if args.mode == "concurrent":
        from concurrent.futures import ThreadPoolExecutor
        import threading
        other = work / "other.onnx"
        other_proto = onnx.load(model)
        other_proto.graph.initializer[0].CopyFrom(
            numpy_helper.from_array(
                np.arange(24, dtype=np.float32).reshape(4, 6) / 7, "w"))
        other_proto.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 6
        onnx.save(other_proto, other)
        expected_other = session(other, ["CPUExecutionProvider"]).run(
            None, {"x": x})[0]
        start = threading.Barrier(2)

        def run_one(path, reference):
            start.wait(timeout=15)
            for _ in range(2):
                actual = session(path, ["AnncExecutionProvider",
                                        "CPUExecutionProvider"]).run(
                    None, {"x": x})[0]
                np.testing.assert_allclose(actual, reference, rtol=1e-3,
                                           atol=1e-7)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_one, model, expected),
                       pool.submit(run_one, other, expected_other)]
            for future in futures:
                future.result()
        entries = list((work / "cache").glob("v1/*/*/DONE"))
        assert len(entries) == 2, entries
        print("Two models compiled and executed concurrently; both numeric")
        return

    # normal: cold compile + warm cache, a second run with a different batch so
    # the context-allocated output size provably follows the runtime input
    # rather than any static report shape, and a batch-0 run whose empty output
    # only the context-allocating kernel can produce.
    for expected_batch in (batch, batch + 3, 0):
        inputs = np.arange(expected_batch * 12, dtype=np.float32).reshape(
            expected_batch, 3, 4) / 11
        reference = session(model, ["CPUExecutionProvider"]).run(
            None, {"x": inputs})[0]
        actual = session(model, ["AnncExecutionProvider", "CPUExecutionProvider"]).run(
            None, {"x": inputs})[0]
        np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-7)
    lines = marker.read_text().splitlines()
    assert len(lines) == 3, lines
    assert lines[0].startswith(f"compute dims={batch},3,5 threads=1 "), lines
    assert lines[1].startswith(f"compute dims={batch + 3},3,5 threads=1 "), lines
    assert lines[2].startswith("compute dims=0,3,5 "), lines
    entries = list((work / "cache").glob("v1/*/*/DONE"))
    assert len(entries) == 1, entries
    assert (entries[0].parent / "libannc_kernels.so").is_file()
    print("Context kernel: allocation + temp + pool; cold compile, warm cache")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolchain-info", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--out-dir")
    parser.add_argument("--ep-library")
    parser.add_argument("--work-dir")
    parser.add_argument("--mode",
                        choices=["normal", "threads", "nested", "concurrent",
                                 "error", "bad-slot"],
                        default="normal")
    args = parser.parse_args()
    if args.toolchain_info or args.model:
        run_pipeline(args)
    else:
        run_test(args)


if __name__ == "__main__":
    main()
