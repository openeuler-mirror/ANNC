#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ADDONS_DIR="${ADDONS_DIR:-${SCRIPT_DIR}/build}"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/session_config.pbtxt}"
BASELINE_CONFIG_FILE="${BASELINE_CONFIG_FILE:-${SCRIPT_DIR}/baseline_session_config.pbtxt}"

MODELS=${1:-"wide_and_deep"}
INTRA_OP=${2:-20}
INTER_OP=${3:-20}
INSTANCE_NUMS=${4:-1}
ENABLE_ANNC=${5:-1}

START_CPU=${START_CPU:-0}

export KDNN_FORCE_NEON=1

# Timeline 开关：ANNC_TIMELINE=1 时把 ANNCFused 各阶段事件写到 trace.json
# 用 chrome://tracing 或 https://ui.perfetto.dev/ 打开
ANNC_TIMELINE=${ANNC_TIMELINE:-0}

# ============================================================================
# 控制参数说明
# ============================================================================
#
# ── 命令行位置参数 ──
#   $1 MODELS          : 模型列表,逗号分隔 (默认: wide_and_deep)
#   $2 INTRA_OP        : TF intra_op 并行度 (默认: 1)
#   $3 INTER_OP        : TF inter_op 并行度 (默认: -1=自动)
#   $4 INSTANCE_NUMS   : serving/client 实例数 (默认: 1)
#   $5 ENABLE_ANNC      : 是否启用 ANNC, 0=baseline (默认: 1)
#
# ── 环境变量 (脚本级控制) ──
#   START_CPU          : NUMA 起始核 (默认: 0)
#   COMPARE_BASELINE   : 是否跑基线 (默认: 1, 当前已禁用)
#   ANNC_PIPELINE_PATH : annc-tf-pipeline 路径
#   ANNC_BACKEND       : ANNC 后端 (默认: generic)
#   ANNC_WORK_BASE     : 工作目录 (默认: /tmp/annc_modelzoo_work)
#   TFSERVER_PATH      : tensorflow_model_server 路径
#   MODEL_BASE         : 模型根目录
#   ADDONS_DIR         : libannc_optimizer.so 目录
#   CONFIG_FILE        : session_config.pbtxt 路径
#
# ── ANNC 运行环境变量 (注入 serving 子进程) ──
#   LD_PRELOAD         : libannc_optimizer.so (自动)
#   ANNC_ENABLE        : 1 (启用 ANNC, 固定)
#   ANNC_VERBOSE       : 1 (verbose 日志, 固定)
#   ANNC_PIPELINE_PATH : annc-tf-pipeline 路径 (同上)
#   ANNC_WORK_DIR      : 模型级临时工作目录 (自动生成)
#   ANNC_SAVEDMODEL_PATH : 模型路径 (自动传入)
#   ANNC_BACKEND       : 后端类型 (同上)
#
# ── 模型压测参数 ──
#   wide_and_deep  : 2:2:2,96
#   dlrm           : 2:2:2,256
#   deepfm         : 2:2:2,256
#   dffm           : 2:2:2,96
#   dssm           : 2:2:2,530
#   格式: concurrency:stop:step,batch
#
# ── 硬编码参数 ──
#   cpu_per_server = 24    : serving 占用核数
#   cpu_per_client = 24    : client 占用核数
#   LD_LIBRARY_PATH        : +tensorflow site-packages
#
# ── 使用示例 ──
#   START_CPU=0 bash test_model_zoo_annc.sh wide_and_deep
#   START_CPU=0 bash test_model_zoo_annc.sh wide_and_deep,dlrm,deepfm 20 20 1 1
#   START_CPU=80 bash test_model_zoo_annc.sh wide_and_deep,dlrm,deepfm
#
# ============================================================================


ANNC_PIPELINE_PATH="${ANNC_PIPELINE_PATH:-/annc/ANNC/ANNC_630/install/bin/annc-tf-pipeline}"
ANNC_BACKEND="${ANNC_BACKEND:-generic}"
ANNC_WORK_BASE="${ANNC_WORK_BASE:-/tmp/annc_modelzoo_work}"
TFSERVER_PATH="${TFSERVER_PATH:-/usr/local/bin/tensorflow_model_server}"
MODEL_BASE="${MODEL_BASE:-/annc/sra_benchmark/modelzoo/}"

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
THREAD_TAG="i${INTRA_OP}-o${INTER_OP}-n${INSTANCE_NUMS}"
if [ "$ENABLE_ANNC" -eq 1 ]; then
    BACKEND_TAG="${ANNC_BACKEND}"
    LOG_TAG="${BACKEND_TAG}_annc"
else
    LOG_TAG="baseline"
fi
LOG_DIR="logs/${LOG_TAG}_modelzoo_${MODELS}_${THREAD_TAG}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/test.log"

declare -A model_params
# model_params["wide_and_deep"]="12:24:1,96"
# model_params["dlrm"]="20:30:1,256"
# model_params["deepfm"]="12:24:1,256"
# model_params["dffm"]="12:24:1,96"
# model_params["dssm"]="12:24:1,530"

# 1 -1 (这些concurrency可以跑满)
# model_params["wide_and_deep"]="10:20:2,96"
# model_params["dlrm"]="18:26:2,256"
# model_params["deepfm"]="16:26:2,256"
# model_params["dffm"]="16:26:2,96"
# model_params["dssm"]="16:26:2,530"

# 1 1
model_params["wide_and_deep"]="12:24:2,96"
model_params["dlrm"]="10:24:2,256"
model_params["deepfm"]="10:24:2,256"
model_params["dffm"]="10:24:2,96"
model_params["dssm"]="10:24:2,530"


RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MAX_RETRIES=3

calculate_numa_node() {
    local cpu=$1
    if [ "$cpu" -lt 80 ]; then
        echo 0
    elif [ "$cpu" -lt 160 ]; then
        echo 1
    elif [ "$cpu" -lt 240 ]; then
        echo 2
    else
        echo 3
    fi
}

check_prerequisites() {
    echo "=========================================="
    echo "ANNC + TF Serving Model Zoo Benchmark"
    echo "=========================================="
    echo ""

    local ok=1

    if [ ! -f "${ADDONS_DIR}/libannc_optimizer.so" ]; then
        echo -e "${RED}ERROR${NC}: libannc_optimizer.so not found at ${ADDONS_DIR}/"
        echo "  Run: cd ${SCRIPT_DIR} && bash build.sh"
        ok=0
    fi

    if [ ! -f "${CONFIG_FILE}" ]; then
        echo -e "${RED}ERROR${NC}: session_config.pbtxt not found at ${CONFIG_FILE}"
        ok=0
    fi

    if [ ! -x "${ANNC_PIPELINE_PATH}" ]; then
        echo -e "${RED}ERROR${NC}: annc-tf-pipeline not found at ${ANNC_PIPELINE_PATH}"
        ok=0
    fi

    if ! command -v "${TFSERVER_PATH}" &>/dev/null && [ ! -x "${TFSERVER_PATH}" ]; then
        echo -e "${RED}ERROR${NC}: tensorflow_model_server not found at ${TFSERVER_PATH}"
        ok=0
    fi

    if ! command -v docker &>/dev/null; then
        echo -e "${YELLOW}WARN${NC}: docker not found, perf_analyzer will not work"
    fi

    if [ "$ok" -eq 0 ]; then
        echo ""
        echo -e "${RED}Prerequisites check failed. Exiting.${NC}"
        exit 1
    fi

    echo -e "${GREEN}Prerequisites OK${NC}"
    echo "  ADDONS_DIR:      ${ADDONS_DIR}"
    echo "  CONFIG_FILE:     ${CONFIG_FILE}"
    echo "  PIPELINE_PATH:   ${ANNC_PIPELINE_PATH}"
    echo "  TFSERVER_PATH:   ${TFSERVER_PATH}"
    echo "  MODEL_BASE:      ${MODEL_BASE}"
    echo "  START_CPU:       ${START_CPU}"
    echo "  INTRA_OP:        ${INTRA_OP}"
    echo "  INTER_OP:        ${INTER_OP}"
    echo "  INSTANCE_NUMS:   ${INSTANCE_NUMS}"
    echo "  LOG_DIR:         ${LOG_DIR}"
    echo ""
}

get_model_dir() {
    local model=$1
    case "$model" in
        "wide_and_deep")
            echo "$MODEL_BASE/wide_and_deep/result/model_WIDE_AND_DEEP/1731296400/"
            ;;
        "dlrm")
            echo "$MODEL_BASE/dlrm/result/model_DLRM/1731305215/"
            ;;
        "dssm")
            echo "$MODEL_BASE/dssm/result/model_DSSM/1730858967/"
            ;;
        "dffm")
            echo "$MODEL_BASE/dffm/result/model_DFFM/1731397983/"
            ;;
        "deepfm")
            echo "$MODEL_BASE/deepfm/result/model_DeepFM/1730800001/"
            ;;
        *)
            echo ""
            return 1
            ;;
    esac
}

start_serving_baseline() {
    local times=$1
    local model=$2
    local tf_model_dir=$3
    local cpu_per_server=20

    for ((i = 0; i < times; i++)); do
        local segment=$(( i % 4 ))
        local cpu_start=$(( segment * cpu_per_server + (i / 4) * 80 + START_CPU ))
        local cpu_end=$(( cpu_start + cpu_per_server - 1 ))
        local numa_node=$(calculate_numa_node "$cpu_start")
        local port=$(( 8888 + i ))
        local rest_port=$(( 8988 + i ))
        local server_log="$LOG_DIR/baseline_serving${i}_${model}.log"

        local subcmd="numactl -C ${cpu_start}-${cpu_end} -m ${numa_node} ${TFSERVER_PATH} --port=${port} --rest_api_port=${rest_port} --model_name=coarse --model_base_path=${tf_model_dir} --tensorflow_session_config_file=${BASELINE_CONFIG_FILE} --tensorflow_intra_op_parallelism=${INTRA_OP} --tensorflow_inter_op_parallelism=${INTER_OP} --enable_profiler=true"

        nohup bash -c "$subcmd" > "$server_log" 2>&1 &
        echo "  baseline server${i}: port=${port} rest=${rest_port} cpus=${cpu_start}-${cpu_end} numa=${numa_node}"
    done
}

start_serving_annc() {
    local times=$1
    local model=$2
    local tf_model_dir=$3
    local cpu_per_server=20

    for ((i = 0; i < times; i++)); do
        local segment=$(( i % 4 ))
        local cpu_start=$(( segment * cpu_per_server + (i / 4) * 80 + START_CPU ))
        local cpu_end=$(( cpu_start + cpu_per_server - 1 ))
        local numa_node=$(calculate_numa_node "$cpu_start")
        local port=$(( 8888 + i ))
        local rest_port=$(( 8988 + i ))
        local server_log="$LOG_DIR/annc_serving${i}_${model}.log"
        local annc_work="${ANNC_WORK_BASE}/${model}_${i}"
        mkdir -p "$annc_work"

        local trace_env=""
        if [ "$ANNC_TIMELINE" -eq 1 ]; then
            local trace_file="$LOG_DIR/trace_annc_${model}_${i}.json"
            trace_env="ANNC_TRACE_FILE=${trace_file}"
            echo "  Timeline: $trace_file"
        fi

        local subcmd="numactl -C ${cpu_start}-${cpu_end} -m ${numa_node} env ${trace_env} PATH=/opt/llvm-21.1.3/install/bin:\${PATH} LD_LIBRARY_PATH=/opt/llvm-21.1.3/install/lib:/usr/local/lib64/python3.11/site-packages/tensorflow:\${LD_LIBRARY_PATH} LD_PRELOAD=${ADDONS_DIR}/libannc_optimizer.so ANNC_ENABLE=1 ANNC_VERBOSE=1 ANNC_PIPELINE_PATH=${ANNC_PIPELINE_PATH} ANNC_WORK_DIR=${annc_work} ANNC_SAVEDMODEL_PATH=${tf_model_dir} ANNC_BACKEND=${ANNC_BACKEND} ANNC_FUSED_PROFILE=${ANNC_FUSED_PROFILE:-0} ANNC_FUSED_PROFILE_INTERVAL=${ANNC_FUSED_PROFILE_INTERVAL:-1000} ${TFSERVER_PATH} --port=${port} --rest_api_port=${rest_port} --model_name=coarse --model_base_path=${tf_model_dir} --tensorflow_session_config_file=${CONFIG_FILE} --tensorflow_intra_op_parallelism=${INTRA_OP} --tensorflow_inter_op_parallelism=${INTER_OP} --enable_profiler=true"

        nohup bash -c "$subcmd" > "$server_log" 2>&1 &
        echo "  ANNC server${i}: port=${port} rest=${rest_port} cpus=${cpu_start}-${cpu_end} numa=${numa_node}"
    done
}

wait_for_serving() {
    local times=$1
    local base_rest_port=${2:-8988}
    local max_wait=${3:-60}

    echo "  Waiting for ${times} serving instance(s) to be ready ..."
    for ((i = 0; i < times; i++)); do
        local rest_port=$(( base_rest_port + i ))
        local ready=0
        for j in $(seq 1 "$max_wait"); do
            if curl -s "http://localhost:${rest_port}/v1/models/coarse" 2>/dev/null | grep -q "AVAILABLE"; then
                ready=1
                break
            fi
            sleep 1
        done
        if [ "$ready" -eq 1 ]; then
            echo -e "  ${GREEN}OK${NC}: instance${i} ready (port ${rest_port})"
        else
            echo -e "  ${RED}FAIL${NC}: instance${i} NOT ready after ${max_wait}s (port ${rest_port})"
            return 1
        fi
    done
    return 0
}

start_client() {
    local times=$1
    local model=$2
    local batch=$3
    local concurrency=$4
    local tag=$5
    local cpu_per_client=20

    client_pids=()
    for ((i = 0; i < times; i++)); do
        local segment=$(( i % 4 ))
        local cpu_start=$(( segment * cpu_per_client + (i / 4) * 80 + START_CPU ))
        local cpu_end=$(( cpu_start + cpu_per_client - 1 ))
        local numa_node=$(calculate_numa_node "$cpu_start")
        local port=$(( 8888 + i ))

        local client_log="$LOG_DIR/client${i}-${model}-${batch}-${tag}.log"
        local subcmd="perf_analyzer --concurrency-range ${concurrency} -p 8000 --latency-threshold 200 -f perf.csv -m coarse --service-kind tfserving -i grpc --request-distribution poisson -b ${batch} -u localhost:${port} --percentile 99 --input-data=random -v"

        docker run --rm --cpuset-cpus="${cpu_start}-${cpu_end}" --cpuset-mems="${numa_node}" --net host nvcr.io/nvidia/tritonserver:24.05-py3-sdk sh -c "$subcmd" > "$client_log" 2>&1 &
        client_pids+=($!)
        echo "client${i} log redirect to $client_log" | tee -a "$LOG_FILE"
    done
}

extract_result() {
    local model=$1
    local batch=$2
    local tag=$3

    local sum=0
    local expr=""
    local files
    files=$(ls "$LOG_DIR"/client*-${model}-${batch}-${tag}.log 2>/dev/null || true)

    for file in $files; do
        local top3
        top3=$(awk -F'[:, ]+' '
        /^Concurrency:/ {
            concurrency = $2
            throughput = $4 + 0
            latency = $7 + 0
            if (latency < 40000) {
                count++
                data[count] = sprintf("%d %d %d", throughput, concurrency, latency)
            }
        }
        END {
            # Skip first/last only if sweep has >=3 levels; with 1-2 levels just print all
            if (count <= 2) {
                for (i = 1; i <= count; i++) print data[i]
            } else {
                for (i = 2; i < count; i++) print data[i]
            }
        }
        ' "$file" | sort -nrk1 | head -n 3)

        if [ -z "$top3" ]; then
            echo "$file: no valid entries (latency >= 40000)"
            continue
        fi

        local i=0
        while read -r tp cc lat; do
            if [ "$i" -eq 0 ]; then
                echo "$file: Throughput: $tp, Concurrency: $cc, Latency: $lat usec (*)"
                sum=$(echo "$sum + $tp" | bc)
                if [ -z "$expr" ]; then
                    expr="$tp"
                else
                    expr="$expr+$tp"
                fi
            else
                echo "$file: Throughput: $tp, Concurrency: $cc, Latency: $lat usec"
            fi
            i=$((i + 1))
        done <<< "$top3"
    done

    echo "=$expr=$sum"
}

stop_serving() {
    pkill -f "tensorflow_model_server" 2>/dev/null || true
    sleep 2
    echo "  All serving processes stopped."
}

cleanup() {
    echo ""
    echo -e "${YELLOW}Interrupted! Cleaning up ...${NC}"
    stop_serving
    echo "  Cleanup done."
    exit 1
}

trap cleanup INT TERM

run_model_test() {
    local model=$1
    local tf_model_dir
    tf_model_dir=$(get_model_dir "$model")

    if [ -z "$tf_model_dir" ]; then
        echo -e "${RED}Unknown model: ${model}${NC}"
        return 1
    fi

    local param_string="${model_params[$model]}"
    IFS=',' read -r -a params <<< "$param_string"
    local concurrency="${params[0]}"
    local batchs="${params[1]}"

    echo "model: $tf_model_dir" | tee -a "$LOG_FILE"
    echo "concurrency: $concurrency" | tee -a "$LOG_FILE"
    echo "batch: $batchs" | tee -a "$LOG_FILE"
    echo "instance nums: $INSTANCE_NUMS" | tee -a "$LOG_FILE"
    echo "start cpu: $START_CPU" | tee -a "$LOG_FILE"

    if [ "$ENABLE_ANNC" -eq 1 ]; then
        local tag="annc"
        echo "start serving (ANNC)..." | tee -a "$LOG_FILE"
        start_serving_annc "$INSTANCE_NUMS" "$model" "$tf_model_dir" | tee -a "$LOG_FILE"
    else
        local tag="baseline"
        echo "start serving (Baseline)..." | tee -a "$LOG_FILE"
        start_serving_baseline "$INSTANCE_NUMS" "$model" "$tf_model_dir" | tee -a "$LOG_FILE"
    fi

    if [ "$ENABLE_ANNC" -eq 1 ]; then
        if ! wait_for_serving "$INSTANCE_NUMS" 8988 90; then
            echo -e "${RED}ANNC serving failed to start${NC}"
            stop_serving
            return 1
        fi
    else
        if ! wait_for_serving "$INSTANCE_NUMS" 8988 60; then
            echo -e "${RED}Baseline serving failed to start${NC}"
            stop_serving
            return 1
        fi
    fi

    IFS='/' read -ra batch_list <<< "$batchs"
    for batch in "${batch_list[@]}"; do
        client_pids=()
        RETRIES=0
        while [ $RETRIES -lt $MAX_RETRIES ]; do
            start_client "$INSTANCE_NUMS" "$model" "$batch" "$concurrency" "$tag"
            echo "Wait all clients..." | tee -a "$LOG_FILE"

            # Capture profiler timeline while clients are running (baseline only, first attempt)
            local profiler_pid=""
            if [ $RETRIES -eq 39 ] && command -v python3 &>/dev/null; then
                echo "  Capturing TF Serving profiler timeline (continuous)..." | tee -a "$LOG_FILE"
                local profiler_log="$LOG_DIR/profiler_${model}_${batch}_${tag}.log"
                # Pass client PIDs so profiler loop knows when to stop
                local pid_file="$LOG_DIR/profiler_pids_${model}_${batch}"
                printf '%s\n' "${client_pids[@]}" > "$pid_file"
                (
                    cd "${SCRIPT_DIR}"
                    LD_LIBRARY_PATH=/usr/local/lib64/python3.11/site-packages/tensorflow \
                    python3 -c "
import tensorflow as tf, os, json, time
from collections import defaultdict
from tensorflow.core.profiler.protobuf import xplane_pb2

outdir = '${LOG_DIR}/profiler_${model}_${batch}'
pid_file = '${pid_file}'
os.makedirs(outdir, exist_ok=True)

def clients_alive():
    try:
        with open(pid_file) as f:
            pids = f.read().strip().split()
        if not pids: return False
        for p in pids:
            try: os.kill(int(p), 0); return True
            except OSError: pass
        return False
    except: return False

segment = 0
all_events = defaultdict(list)  # name -> [(ts_us, dur_us, tid), ...]
all_names = set()
accumulated_end = 0  # us offset for chaining segments

while True:
    if not clients_alive() or segment >= 2:
        print('Profiler stopping.')
        break

    seg_dir = os.path.join(outdir, f'seg{segment}')
    os.makedirs(seg_dir, exist_ok=True)
    print(f'Trace segment {segment} (30s)...')
    try:
        tf.profiler.experimental.client.trace('localhost:8888', seg_dir, 30000, num_tracing_attempts=2)
    except Exception as e:
        print(f'Trace failed: {e}')
        break

    pb = None
    for root, dirs, files in os.walk(seg_dir):
        for f in files:
            if f.endswith('.xplane.pb'):
                pb = os.path.join(root, f); break
    if not pb or os.path.getsize(pb) < 1000:
        print('Empty xplane, stopping.')
        break

    xs = xplane_pb2.XSpace()
    with open(pb, 'rb') as fh: xs.ParseFromString(fh.read())

    # Collect events with timestamps relative to this segment
    seg_events = []
    tids_seen = set()
    for plane in xs.planes:
        meta_map = {k: v.name for k, v in plane.event_metadata.items()}
        for line in plane.lines:
            tid = line.id  # stream/thread id
            for evt in line.events:
                name = meta_map.get(evt.metadata_id, '')
                all_names.add(name)
                # Only capture MatMul / ANNCFused operator events
                if 'matmul' not in name.lower() and 'anncfused' not in name.lower():
                    continue
                dur_us = evt.duration_ps / 1e6
                if dur_us <= 0:
                    continue
                ts_us = evt.offset_ps / 1e6  # relative to this trace start
                seg_events.append((ts_us, dur_us, name, tid))
                tids_seen.add(tid)

    os.remove(pb)

    # Sort by timestamp within segment
    seg_events.sort(key=lambda x: x[0])

    # Append with accumulated offset so events chain across segments
    segment_start = seg_events[0][0] if seg_events else 0
    n = 0
    all_tids = set()
    for ts_us, dur_us, name, tid in seg_events:
        rel_ts = ts_us - segment_start + accumulated_end
        all_events[name].append((rel_ts, dur_us, tid))
        all_tids.add(tid)
        n += 1
    # Move accumulated_end forward: end of last event in this segment + small gap
    if seg_events:
        last_ts = seg_events[-1][0] - segment_start
        last_dur = seg_events[-1][1]
        accumulated_end += last_ts + last_dur + 100  # 100us gap between segments

    print(f'  {n} events')
    segment += 1

matmul_names = sorted(n for n in all_names if 'matmul' in n.lower() or 'anncfused' in n.lower())
print(f'Segments: {segment}, all ops: {len(all_names)}, MatMul/ANNCFused: {len(matmul_names)}')

if all_events:
    total_events = sum(len(v) for v in all_events.values())
    # Build flat lists for summary (durations only); collect all tids for metadata
    dur_by_name = {}
    for name, events in all_events.items():
        dur_by_name[name] = [d for _, d, _ in events]
    total_us = sum(sum(v) for v in dur_by_name.values())

    # Collect unique tids across all events
    all_tids = set()
    for events in all_events.values():
        for _, _, tid in events:
            all_tids.add(tid)

    json_path = os.path.join(outdir, 'timeline.json')
    with open(json_path, 'w') as f:
        f.write('[\n')
        # Metadata: thread_name entries so chrome://tracing shows labelled lanes
        first = True
        for tid in sorted(all_tids):
            if not first: f.write(',\n')
            f.write(json.dumps({'ph': 'M', 'pid': 0, 'tid': tid, 'name': 'thread_name', 'args': {'name': f'Stream {tid}'}}, separators=(',', ':')))
            first = False
        # Complete events
        for name in sorted(all_events.keys()):
            for ts, d, tid in all_events[name]:
                if not first: f.write(',\n')
                f.write(json.dumps({'name': name, 'ph': 'X', 'pid': 0, 'tid': tid, 'ts': ts, 'dur': d}, separators=(',', ':')))
                first = False
        f.write('\n]\n')
    print(f'JSON: {os.path.getsize(json_path)} bytes')

    sp = os.path.join(outdir, 'timeline_summary.txt')
    with open(sp, 'w') as f:
        f.write(f'Operator timing ({total_events} events, {segment} segments)\n')
        f.write(f'{\"Op\":<70} {\"Count\":>7} {\"Avg(us)\":>9} {\"Min(us)\":>9} {\"Max(us)\":>9} {\"Total(ms)\":>10} {\"%\":>6}\n')
        f.write('-'*112 + '\n')
        for name in sorted(dur_by_name.keys()):
            durs = dur_by_name[name]
            t = sum(durs) / 1000
            pct = t / max(total_us / 1000, 1) * 100
            f.write(f'{name:<70} {len(durs):>7} {sum(durs)/len(durs):>9.1f} {min(durs):>9.1f} {max(durs):>9.1f} {t:>10.1f} {pct:>5.1f}%\n')
        f.write('-'*112 + '\n')
        f.write(f'{\"TOTAL\":<70} {\"\":>7} {\"\":>9} {\"\":>9} {\"\":>9} {total_us/1000:>10.1f}\n')
    print(f'Summary: {sp}')
else:
    print('No events captured')

os.remove(pid_file)
" > "$profiler_log" 2>&1
                ) &
                profiler_pid=$!
                echo "  Profiler PID=${profiler_pid}, log=$profiler_log" | tee -a "$LOG_FILE"
            fi

            EXIT_CODE=0
            for pid in "${client_pids[@]}"; do
                wait "$pid" && true || EXIT_CODE=1
            done

            # Wait for profiler to finish if running
            if [ -n "$profiler_pid" ]; then
                wait "$profiler_pid" 2>/dev/null || true
                echo "  Profiler capture finished" | tee -a "$LOG_FILE"
            fi

            if [ $EXIT_CODE -eq 0 ]; then
                break
            else
                RETRIES=$((RETRIES + 1))
                client_pids=()
                sleep 3
            fi
        done
        echo "extract result..." | tee -a "$LOG_FILE"
        echo "------------ batch: $batch ------------------" | tee -a "$LOG_FILE"
        extract_result "$model" "$batch" "$tag" | tee -a "$LOG_FILE"
    done

    if [ "$ENABLE_ANNC" -eq 1 ]; then
        check_annc_graph "$model"
    fi
    echo "Stopping all servers..." | tee -a "$LOG_FILE"
    stop_serving
    echo ""
}

check_annc_graph() {
    local model=$1

    echo "  --- ANNC Graph Verification ---"
    local annc_count=0
    for log in "$LOG_DIR"/annc_serving*_${model}.log; do
        if [ -f "$log" ]; then
            local count
            count=$(grep -c "ANNCFused\|graph rewrite completed\|annc_fused" "$log" 2>/dev/null || echo 0)
            annc_count=$((annc_count + count))
        fi
    done

    if [ "$annc_count" -gt 0 ]; then
        echo -e "  ${GREEN}PASS${NC}: ANNC graph rewrite detected (${annc_count} matches)"
    else
        echo -e "  ${RED}FAIL${NC}: No ANNC graph rewrite detected in serving logs"
    fi

    local kernel_count=0
    for log in "$LOG_DIR"/annc_serving*_${model}.log; do
        if [ -f "$log" ]; then
            local count
            count=$(grep -c 'annc_generated_kernel\|fused op library' "$log" 2>/dev/null || echo 0)
            kernel_count=$((kernel_count + count))
        fi
    done

    if [ "$kernel_count" -gt 0 ]; then
        echo -e "  ${GREEN}PASS${NC}: ANNC kernel .so detected (${kernel_count} matches)"
    else
        echo -e "  ${YELLOW}WARN${NC}: No ANNC kernel .so reference found"
    fi
}

check_prerequisites

IFS=',' read -ra models_array <<< "$MODELS"

for model in "${models_array[@]}"; do
    run_model_test "$model"
done

echo ""
echo "=========================================="
echo "All tests completed."
echo "Logs: ${LOG_DIR}/"
echo "=========================================="
