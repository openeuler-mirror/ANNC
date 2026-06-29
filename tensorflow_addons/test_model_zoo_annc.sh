#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_ANNC_INSTALL="${PROJECT_ROOT}/install"
DEFAULT_ANNC_BUILD="${PROJECT_ROOT}/build"

if [ -x "${DEFAULT_ANNC_INSTALL}/bin/annc-tf-pipeline" ]; then
    DEFAULT_ANNC_BIN="${DEFAULT_ANNC_INSTALL}/bin"
else
    DEFAULT_ANNC_BIN="${DEFAULT_ANNC_BUILD}/bin"
fi

if [ -f "${DEFAULT_ANNC_INSTALL}/lib/libannc_optimizer.so" ]; then
    DEFAULT_ADDONS_DIR="${DEFAULT_ANNC_INSTALL}/lib"
else
    DEFAULT_ADDONS_DIR="${DEFAULT_ANNC_BUILD}/lib"
fi

export PATH="${DEFAULT_ANNC_BIN}:$PATH"

ADDONS_DIR="${ADDONS_DIR:-${DEFAULT_ADDONS_DIR}}"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/session_config.pbtxt}"
BASELINE_CONFIG_FILE="${BASELINE_CONFIG_FILE:-${SCRIPT_DIR}/baseline_session_config.pbtxt}"

MODELS=${1:-"wide_and_deep"}
INTRA_OP=${2:-24}
INTER_OP=${3:-24}
INSTANCE_NUMS=${4:-1}
ENABLE_ANNC=${5:-1}

# Backward compatibility: START_CPU is the server-side anchor.  Use SERVER_START_CPU
# when you want to place the client elsewhere explicitly.
START_CPU=${START_CPU:-0}
SERVER_START_CPU=${SERVER_START_CPU:-$START_CPU}

BASE_PORT=${BASE_PORT:-8888}
BASE_REST_PORT=$((BASE_PORT + 100))

SERVER_PIDS=()

# By default the client is placed on a different NUMA node than the server so that
# client and server do not contend for the same CPU cores.  This rule is applied
# identically to baseline and ANNc runs to keep the comparison fair.
CLIENT_NUMA_NODE=${CLIENT_NUMA_NODE:-}
CLIENT_START_CPU=${CLIENT_START_CPU:-}

#export KDNN_FORCE_NEON=1

# 远程 ANNC 仓库当前未实现 ANNC_TRACE_FILE timeline 注入，默认不启用。
ANNC_TIMELINE=${ANNC_TIMELINE:-0}

# Profiler 开关：ENABLE_PROFILER=1 时启用 TF Serving profiler 服务并捕获 timeline
ENABLE_PROFILER=${ENABLE_PROFILER:-0}
if [ "$ENABLE_PROFILER" -eq 1 ]; then
    TF_PROFILER_FLAG="true"
else
    TF_PROFILER_FLAG="false"
fi

# 是否在一次调用中顺序跑 baseline + ANNc 并生成对比日志。
# 默认开启；设置为 0 时回到旧行为，只跑 ENABLE_ANNC 指定的模式。
RUN_BOTH=${RUN_BOTH:-1}

# perf_analyzer 延迟阈值（单位：微秒）。推荐模型 batch 较大，200 us 过严，
# 默认放宽到 200 ms，可通过环境变量覆盖。
LATENCY_THRESHOLD=${LATENCY_THRESHOLD:-200000}

# perf_analyzer 正式测量前发送的 warm-up 请求数，用于触发 ANNc 图改写 / kernel 编译。
# 设为 0 可跳过 warm-up。
WARMUP_REQUESTS=${WARMUP_REQUESTS:-50}

# ============================================================================
# 控制参数说明
# ============================================================================
#
# ── 命令行位置参数 ──
#   $1 MODELS          : 模型列表,逗号分隔 (默认: wide_and_deep)
#   $2 INTRA_OP        : TF intra_op 并行度 (默认: 24)
#   $3 INTER_OP        : TF inter_op 并行度 (默认: 24)
#   $4 INSTANCE_NUMS   : serving/client 实例数 (默认: 1)
#   $5 ENABLE_ANNC      : 是否启用 ANNC, 0=baseline, 1=ANNC (默认: 1)
#                        仅在 RUN_BOTH=0 时生效
#
# ── 环境变量 (脚本级控制) ──
#   START_CPU          : server 起始核 (默认: 0, 兼容旧用法)
#   SERVER_START_CPU   : server 起始核 (默认: START_CPU)
#   CLIENT_NUMA_NODE   : client 使用的 NUMA node (默认: server_node+1)
#   CLIENT_START_CPU   : client 起始核 (默认: CLIENT_NUMA_NODE 第一个核)
#   RUN_BOTH           : 是否一次调用跑 baseline + ANNc (默认: 1)
#   LATENCY_THRESHOLD  : perf_analyzer 延迟阈值 us (默认: 200000)
#   WARMUP_REQUESTS    : perf_analyzer warm-up 请求数 (默认: 50)
#   ANNC_PIPELINE_PATH : annc-tf-pipeline 路径 (默认: 自动推导到本仓库 install/bin)
#   LLVM_PATH          : LLVM 安装目录 (默认: 从 ANNC_PIPELINE_PATH 推导)
#   TF_PY_SITE_PACKAGES: TensorFlow Python site-packages 目录 (默认: 自动检测)
#   ANNC_BACKEND       : ANNC 后端 (默认: generic)
#   ANNC_WORK_BASE     : 工作目录 (默认: /tmp/annc_modelzoo_work)
#   TFSERVER_PATH      : tensorflow_model_server 路径 (默认: 自动检测)
#   MODEL_BASE         : 模型根目录 (默认: 自动检测)
#   ADDONS_DIR         : libannc_optimizer.so 目录
#   CONFIG_FILE        : session_config.pbtxt 路径
#
# ── ANNC 运行环境变量 (注入 serving 子进程) ──
#   LD_PRELOAD         : libannc_optimizer.so (自动)
#   ANNC_ENABLE        : 1 (启用 ANNC, 固定)
#   ANNC_VERBOSE       : 0 (默认关闭 verbose 日志，避免污染性能数据)
#   ANNC_PIPELINE_PATH : annc-tf-pipeline 路径 (同上)
#   ANNC_WORK_DIR      : 模型级临时工作目录 (自动生成)
#   ANNC_SAVEDMODEL_PATH : 模型路径 (自动传入)
#   ANNC_BACKEND       : 后端类型 (同上)
#
# ── 模型压测参数 ──
#   wide_and_deep  : 10:24:2,96
#   dlrm           : 10:24:2,256
#   deepfm         : 10:24:2,256
#   dffm           : 10:24:2,96
#   dssm           : 10:24:2,530
#   格式: start:stop:step,batch
#
# ── 硬编码参数 ──
#   cpu_per_server = 24    : serving 占用核数
#   cpu_per_client = 24    : client 占用核数
#   NUMA 拓扑              : 自动从 /sys/devices/system/node 检测
#
# ── 使用示例 ──
#   bash test_model_zoo_annc.sh wide_and_deep
#   bash test_model_zoo_annc.sh wide_and_deep,dlrm,deepfm 20 20 1
#   SERVER_START_CPU=0 CLIENT_START_CPU=48 bash test_model_zoo_annc.sh wide_and_deep
#   RUN_BOTH=0 ENABLE_ANNC=0 bash test_model_zoo_annc.sh wide_and_deep
#
# ============================================================================

DEFAULT_TFSERVER="/home/l00562880/annc/serving/output/826723b62a7b3d9761abfbfce744e16b/execroot/tf_serving/bazel-out/aarch64-opt/bin/tensorflow_serving/model_servers/tensorflow_model_server"
DEFAULT_MODEL_BASE="/home/l00562880/annc/sra_benchmark/modelzoo"

ANNC_PIPELINE_PATH="${ANNC_PIPELINE_PATH:-${DEFAULT_ANNC_BIN}/annc-tf-pipeline}"
ANNC_BACKEND="${ANNC_BACKEND:-generic}"
ANNC_WORK_BASE="${ANNC_WORK_BASE:-/tmp/annc_modelzoo_work}"
TFSERVER_PATH="${TFSERVER_PATH:-${DEFAULT_TFSERVER}}"
MODEL_BASE="${MODEL_BASE:-${DEFAULT_MODEL_BASE}}"
ANNC_FUSED_OP_PATH="${ANNC_FUSED_OP_PATH:-${ADDONS_DIR}/libannc_fused_op.so}"

# Derive LLVM install from the ANNC pipeline location so we don't hard-code /opt/llvm-21.1.3.
ANNC_INSTALL_BIN="$(dirname "$ANNC_PIPELINE_PATH")"
ANNC_INSTALL_ROOT="$(dirname "$ANNC_INSTALL_BIN")"
LLVM_PATH="${LLVM_PATH:-${ANNC_INSTALL_ROOT}}"

# Auto-detect TensorFlow Python site-packages path.
TF_PY_SITE_PACKAGES="${TF_PY_SITE_PACKAGES:-$(python3 -c 'import tensorflow as tf, os; print(os.path.dirname(tf.__file__))' 2>/dev/null || echo "/usr/local/lib64/python3.11/site-packages/tensorflow")}"

# Detect actual NUMA topology from /sys instead of assuming 4 nodes x 80 cores.
NUMA_NODE_CPUS=()
NUMA_NODES=0
CORES_PER_NODE=0

detect_numa_topology() {
    local nodes
    nodes=$(ls -d /sys/devices/system/node/node* 2>/dev/null | sort -V)
    NUMA_NODES=$(echo "$nodes" | wc -l)
    for node_dir in $nodes; do
        local cpulist
        cpulist=$(cat "${node_dir}/cpulist" 2>/dev/null || echo "")
        NUMA_NODE_CPUS+=("$cpulist")
    done
    if [ "$NUMA_NODES" -gt 0 ]; then
        local first_range
        first_range=${NUMA_NODE_CPUS[0]}
        # Count cores in the first range (e.g., "0-47" -> 48)
        local start end
        start=$(echo "$first_range" | cut -d'-' -f1)
        end=$(echo "$first_range" | cut -d'-' -f2)
        CORES_PER_NODE=$((end - start + 1))
    fi
}

detect_numa_topology

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
THREAD_TAG="i${INTRA_OP}-o${INTER_OP}-n${INSTANCE_NUMS}"

# 顶层日志目录：一次调用中 baseline 与 ANNc 的结果放在同一父目录下，便于对比。
BASE_LOG_DIR="logs/modelzoo_${MODELS}_${THREAD_TAG}_${TIMESTAMP}"
mkdir -p "$BASE_LOG_DIR"

declare -A model_params
# 1 1
model_params["wide_and_deep"]="22:28:1,96"
model_params["dlrm"]="44:54:1,256"
model_params["deepfm"]="24:34:1,256"
model_params["dffm"]="26:36:1,96"
model_params["dssm"]="32:37:1,530"
# model_params["dssm"]=":10:1,530"


RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MAX_RETRIES=3

calculate_numa_node() {
    local cpu=$1
    if [ "$CORES_PER_NODE" -le 0 ]; then
        echo 0
        return
    fi
    local node=$(( cpu / CORES_PER_NODE ))
    if [ "$node" -lt 0 ]; then
        node=0
    elif [ "$node" -ge "$NUMA_NODES" ]; then
        node=$((NUMA_NODES - 1))
    fi
    echo "$node"
}

# 获取指定 NUMA node 的第一个 CPU；若不可用则回退到 SERVER_START_CPU
get_numa_node_start_cpu() {
    local node=$1
    if [ "$node" -lt 0 ] || [ "$node" -ge "$NUMA_NODES" ] || [ -z "${NUMA_NODE_CPUS[$node]:-}" ]; then
        echo "$SERVER_START_CPU"
        return
    fi
    local cpulist="${NUMA_NODE_CPUS[$node]}"
    echo "$cpulist" | awk -F'[-,]' '{print $1}'
}

# Auto-place client on a different NUMA node than the server by default so that
# baseline and ANNc runs share the same client/server NUMA topology.
if [ "$NUMA_NODES" -gt 1 ] && [ -z "${CLIENT_START_CPU:-}" ] && [ -z "${CLIENT_NUMA_NODE:-}" ]; then
    server_numa=$(calculate_numa_node "$SERVER_START_CPU")
    CLIENT_NUMA_NODE=$(( (server_numa + 1) % NUMA_NODES ))
fi
if [ -z "${CLIENT_START_CPU:-}" ]; then
    CLIENT_START_CPU=$(get_numa_node_start_cpu "${CLIENT_NUMA_NODE:-0}")
fi

check_prerequisites() {
    echo "=========================================="
    echo "ANNC + TF Serving Model Zoo Benchmark"
    echo "=========================================="
    echo ""

    local ok=1

    if [ ! -f "${ADDONS_DIR}/libannc_optimizer.so" ]; then
        echo -e "${RED}ERROR${NC}: libannc_optimizer.so not found at ${ADDONS_DIR}/"
        echo "  Expected top-level ANNC build artifact under ${PROJECT_ROOT}/install/lib or ${PROJECT_ROOT}/build/lib"
        ok=0
    fi

    if [ ! -f "${ANNC_FUSED_OP_PATH}" ]; then
        echo -e "${RED}ERROR${NC}: libannc_fused_op.so not found at ${ANNC_FUSED_OP_PATH}"
        ok=0
    fi

    if [ ! -f "${CONFIG_FILE}" ]; then
        echo -e "${RED}ERROR${NC}: session_config.pbtxt not found at ${CONFIG_FILE}"
        ok=0
    fi

    if [ ! -f "${BASELINE_CONFIG_FILE}" ]; then
        echo -e "${RED}ERROR${NC}: baseline_session_config.pbtxt not found at ${BASELINE_CONFIG_FILE}"
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
    echo "  FUSED_OP_PATH:   ${ANNC_FUSED_OP_PATH}"
    echo "  CONFIG_FILE:     ${CONFIG_FILE}"
    echo "  BASELINE_CONFIG: ${BASELINE_CONFIG_FILE}"
    echo "  PIPELINE_PATH:   ${ANNC_PIPELINE_PATH}"
    echo "  LLVM_PATH:       ${LLVM_PATH}"
    echo "  TFSERVER_PATH:   ${TFSERVER_PATH}"
    echo "  MODEL_BASE:      ${MODEL_BASE}"
    echo "  TF_PY_PKGS:      ${TF_PY_SITE_PACKAGES}"
    echo "  NUMA_NODES:      ${NUMA_NODES}"
    echo "  CORES_PER_NODE:  ${CORES_PER_NODE}"
    echo "  SERVER_START_CPU:${SERVER_START_CPU}"
    echo "  CLIENT_NUMA_NODE:${CLIENT_NUMA_NODE:-<auto>}"
    echo "  CLIENT_START_CPU:${CLIENT_START_CPU}"
    echo "  INTRA_OP:        ${INTRA_OP}"
    echo "  INTER_OP:        ${INTER_OP}"
    echo "  INSTANCE_NUMS:   ${INSTANCE_NUMS}"
    echo "  RUN_BOTH:        ${RUN_BOTH}"
    echo "  LATENCY_THRESH:  ${LATENCY_THRESHOLD} us"
    echo "  WARMUP_REQUESTS: ${WARMUP_REQUESTS}"
    echo "  BASE_LOG_DIR:    ${BASE_LOG_DIR}"

    local server_numa client_numa
    server_numa=$(calculate_numa_node "$SERVER_START_CPU")
    client_numa=$(calculate_numa_node "$CLIENT_START_CPU")
    if [ "$server_numa" -eq "$client_numa" ]; then
        echo -e "${YELLOW}WARN${NC}: client and server are on the same NUMA node ${server_numa}."
        echo "  Consider setting CLIENT_START_CPU or CLIENT_NUMA_NODE to avoid CPU contention."
    fi
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

SERVER_PIDS=()

kill_port_occupiers() {
    for ((i = 0; i < INSTANCE_NUMS; i++)); do
        local port=$((BASE_PORT + i))
        local rest_port=$((BASE_REST_PORT + i))
        for p in $port $rest_port; do
            local pid
            pid=$(lsof -ti :$p 2>/dev/null || true)
            if [ -n "$pid" ]; then
                echo "  Port $p occupied by PID(s) $pid, killing ..."
                kill $pid 2>/dev/null || true
                sleep 1
                pid=$(lsof -ti :$p 2>/dev/null || true)
                if [ -n "$pid" ]; then
                    kill -9 $pid 2>/dev/null || true
                fi
            fi
        done
    done
}

start_serving_baseline() {
    local times=$1
    local model=$2
    local tf_model_dir=$3
    local cpu_per_server=24
    local instances_per_node=$(( CORES_PER_NODE / cpu_per_server ))
    [ "$instances_per_node" -lt 1 ] && instances_per_node=1

    for ((i = 0; i < times; i++)); do
        local segment=$(( i % instances_per_node ))
        local cpu_start=$(( segment * cpu_per_server + (i / instances_per_node) * CORES_PER_NODE + SERVER_START_CPU ))
        local cpu_end=$(( cpu_start + cpu_per_server - 1 ))
        local numa_node=$(calculate_numa_node "$cpu_start")
        local port=$(( BASE_PORT + i ))
        local rest_port=$(( BASE_REST_PORT + i ))
        local server_log="$LOG_DIR/baseline_serving${i}_${model}.log"

        echo "[FINAL CMD baseline server${i}] numactl -C ${cpu_start}-${cpu_end} -m ${numa_node} ${TFSERVER_PATH} --port=${port} --rest_api_port=${rest_port} --model_name=coarse --model_base_path=${tf_model_dir} --tensorflow_session_config_file=${BASELINE_CONFIG_FILE} --tensorflow_intra_op_parallelism=${INTRA_OP} --tensorflow_inter_op_parallelism=${INTER_OP} --enable_profiler=${TF_PROFILER_FLAG}"
        nohup numactl -C ${cpu_start}-${cpu_end} -m ${numa_node} ${TFSERVER_PATH} --port=${port} --rest_api_port=${rest_port} --model_name=coarse --model_base_path=${tf_model_dir} --tensorflow_session_config_file=${BASELINE_CONFIG_FILE} --tensorflow_intra_op_parallelism=${INTRA_OP} --tensorflow_inter_op_parallelism=${INTER_OP} --enable_profiler=${TF_PROFILER_FLAG} > "$server_log" 2>&1 &
        SERVER_PIDS+=($!)
        echo "  baseline server${i}: pid=$! port=${port} rest=${rest_port} cpus=${cpu_start}-${cpu_end} numa=${numa_node}"
    done
}

start_serving_annc() {
    local times=$1
    local model=$2
    local tf_model_dir=$3
    local cpu_per_server=24
    local instances_per_node=$(( CORES_PER_NODE / cpu_per_server ))
    [ "$instances_per_node" -lt 1 ] && instances_per_node=1

    for ((i = 0; i < times; i++)); do
        local segment=$(( i % instances_per_node ))
        local cpu_start=$(( segment * cpu_per_server + (i / instances_per_node) * CORES_PER_NODE + SERVER_START_CPU ))
        local cpu_end=$(( cpu_start + cpu_per_server - 1 ))
        local numa_node=$(calculate_numa_node "$cpu_start")
        local port=$(( BASE_PORT + i ))
        local rest_port=$(( BASE_REST_PORT + i ))
        local server_log="$LOG_DIR/annc_serving${i}_${model}.log"
        local annc_work="${ANNC_WORK_BASE}/${model}_${i}"
        mkdir -p "$annc_work"

        local trace_env=""
        if [ "$ANNC_TIMELINE" -eq 1 ]; then
            echo -e "  ${YELLOW}WARN${NC}: ANNC_TIMELINE is not supported by this ANNC checkout; ignoring."
        fi

        echo "[FINAL CMD ANNC server${i}] numactl -C ${cpu_start}-${cpu_end} -m ${numa_node} env ${trace_env} PATH=${LLVM_PATH}/bin:\$PATH LD_LIBRARY_PATH=${LLVM_PATH}/lib:${TF_PY_SITE_PACKAGES}:\$LD_LIBRARY_PATH LD_PRELOAD=${ADDONS_DIR}/libannc_optimizer.so ANNC_ENABLE=1 ANNC_VERBOSE=${ANNC_VERBOSE:-1} ANNC_PIPELINE_PATH=${ANNC_PIPELINE_PATH} ANNC_WORK_DIR=${annc_work} ANNC_SAVEDMODEL_PATH=${tf_model_dir} ANNC_BACKEND=${ANNC_BACKEND} ANNC_FUSED_OP_PATH=${ANNC_FUSED_OP_PATH} ${TFSERVER_PATH} --port=${port} --rest_api_port=${rest_port} --model_name=coarse --model_base_path=${tf_model_dir} --tensorflow_session_config_file=${CONFIG_FILE} --tensorflow_intra_op_parallelism=${INTRA_OP} --tensorflow_inter_op_parallelism=${INTER_OP} --enable_profiler=${TF_PROFILER_FLAG}"
        nohup numactl -C ${cpu_start}-${cpu_end} -m ${numa_node} env ${trace_env} PATH=${LLVM_PATH}/bin:${PATH} LD_LIBRARY_PATH=${LLVM_PATH}/lib:${TF_PY_SITE_PACKAGES}:${LD_LIBRARY_PATH:-} LD_PRELOAD=${ADDONS_DIR}/libannc_optimizer.so ANNC_ENABLE=1 ANNC_VERBOSE=${ANNC_VERBOSE:-1} ANNC_PIPELINE_PATH=${ANNC_PIPELINE_PATH} ANNC_WORK_DIR=${annc_work} ANNC_SAVEDMODEL_PATH=${tf_model_dir} ANNC_BACKEND=${ANNC_BACKEND} ANNC_FUSED_OP_PATH=${ANNC_FUSED_OP_PATH} ${TFSERVER_PATH} --port=${port} --rest_api_port=${rest_port} --model_name=coarse --model_base_path=${tf_model_dir} --tensorflow_session_config_file=${CONFIG_FILE} --tensorflow_intra_op_parallelism=${INTRA_OP} --tensorflow_inter_op_parallelism=${INTER_OP} --enable_profiler=${TF_PROFILER_FLAG} > "$server_log" 2>&1 &
        SERVER_PIDS+=($!)
        echo "  ANNC server${i}: pid=$! port=${port} rest=${rest_port} cpus=${cpu_start}-${cpu_end} numa=${numa_node}"
    done
}

wait_for_serving() {
    local times=$1
    local base_rest_port=${2:-$BASE_REST_PORT}
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

warmup_client() {
    local times=$1
    local model=$2
    local batch=$3
    local tag=$4
    local cpu_per_client=24
    local instances_per_node=$(( CORES_PER_NODE / cpu_per_client ))
    [ "$instances_per_node" -lt 1 ] && instances_per_node=1

    # client 与 server 使用对称的绑定策略，避免 baseline 与 ANNc 运行环境不同。
    local client_start_cpu=$CLIENT_START_CPU
    local pids=()

    echo "  Warm-up (${WARMUP_REQUESTS} requests per instance, tag=${tag}) ..." | tee -a "$LOG_FILE"
    for ((i = 0; i < times; i++)); do
        local segment=$(( i % instances_per_node ))
        local cpu_start=$(( segment * cpu_per_client + (i / instances_per_node) * CORES_PER_NODE + client_start_cpu ))
        local cpu_end=$(( cpu_start + cpu_per_client - 1 ))
        local numa_node=$(calculate_numa_node "$cpu_start")
        local port=$(( BASE_PORT + i ))

        local warmup_log="$LOG_DIR/warmup${i}-${model}-${batch}-${tag}.log"
        local subcmd="perf_analyzer --request-count ${WARMUP_REQUESTS} --concurrency-range 1 -p 2000 --latency-threshold ${LATENCY_THRESHOLD} -m coarse --service-kind tfserving -i grpc --request-distribution poisson -b ${batch} -u localhost:${port} --input-data=random -v"

        docker run --rm --cpuset-cpus="${cpu_start}-${cpu_end}" --cpuset-mems="${numa_node}" --net host nvcr.io/nvidia/tritonserver:24.05-py3-sdk sh -c "$subcmd" > "$warmup_log" 2>&1 &
        pids+=($!)
    done

    local exit_code=0
    for pid in "${pids[@]}"; do
        wait "$pid" && true || exit_code=1
    done
    if [ $exit_code -ne 0 ]; then
        echo -e "  ${YELLOW}WARN${NC}: warm-up failed (non-fatal), continuing with real measurement ..." | tee -a "$LOG_FILE"
    fi
}

start_client() {
    local times=$1
    local model=$2
    local batch=$3
    local concurrency=$4
    local tag=$5
    local cpu_per_client=24
    local instances_per_node=$(( CORES_PER_NODE / cpu_per_client ))
    [ "$instances_per_node" -lt 1 ] && instances_per_node=1

    # client 与 server 使用对称的绑定策略，避免 baseline 与 ANNc 运行环境不同。
    local client_start_cpu=$CLIENT_START_CPU

    client_pids=()
    for ((i = 0; i < times; i++)); do
        local segment=$(( i % instances_per_node ))
        local cpu_start=$(( segment * cpu_per_client + (i / instances_per_node) * CORES_PER_NODE + client_start_cpu ))
        local cpu_end=$(( cpu_start + cpu_per_client - 1 ))
        local numa_node=$(calculate_numa_node "$cpu_start")
        local port=$(( BASE_PORT + i ))

        local client_log="$LOG_DIR/client${i}-${model}-${batch}-${tag}.log"
        local subcmd="perf_analyzer --concurrency-range ${concurrency} -p 8000 --latency-threshold ${LATENCY_THRESHOLD} -f perf.csv -m coarse --service-kind tfserving -i grpc --request-distribution poisson -b ${batch} -u localhost:${port} --percentile 99 --input-data=random -v"

        docker run --rm --cpuset-cpus="${cpu_start}-${cpu_end}" --cpuset-mems="${numa_node}" --net host nvcr.io/nvidia/tritonserver:24.05-py3-sdk sh -c "$subcmd" > "$client_log" 2>&1 &
        client_pids+=($!)
        echo "client${i} (tag=${tag}) cpus=${cpu_start}-${cpu_end} numa=${numa_node} log=$client_log" | tee -a "$LOG_FILE"
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
    if [ ${#SERVER_PIDS[@]} -gt 0 ]; then
        echo "  Stopping ${#SERVER_PIDS[@]} tracked server(s): ${SERVER_PIDS[*]}"
        for pid in "${SERVER_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
                echo "    Killed PID $pid"
            else
                echo "    PID $pid already exited"
            fi
        done
        sleep 2
        for pid in "${SERVER_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
                echo "    Force-killed PID $pid"
            fi
        done
        SERVER_PIDS=()
    fi

    echo "  Sweeping port occupants ..."
    local found=0
    for ((i = 0; i < INSTANCE_NUMS; i++)); do
        for p in $((BASE_PORT + i)) $((BASE_REST_PORT + i)); do
            local pids
            pids=$(lsof -ti :$p 2>/dev/null || true)
            if [ -n "$pids" ]; then
                for pid in $pids; do
                    kill -9 "$pid" 2>/dev/null || true
                    echo "    Killed port $p occupant PID $pid"
                    found=1
                done
            fi
        done
    done
    if [ "$found" -eq 0 ]; then
        echo "    No port occupants found."
    fi
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
    echo "server start cpu: $SERVER_START_CPU" | tee -a "$LOG_FILE"
    echo "client start cpu: $CLIENT_START_CPU" | tee -a "$LOG_FILE"

    if [ "$ENABLE_ANNC" -eq 1 ]; then
        local tag="annc"
        echo "start serving (ANNC) ..." | tee -a "$LOG_FILE"
        kill_port_occupiers
        start_serving_annc "$INSTANCE_NUMS" "$model" "$tf_model_dir" > >(tee -a "$LOG_FILE")
    else
        local tag="baseline"
        echo "start serving (Baseline) ..." | tee -a "$LOG_FILE"
        kill_port_occupiers
        start_serving_baseline "$INSTANCE_NUMS" "$model" "$tf_model_dir" > >(tee -a "$LOG_FILE")
    fi

    if [ "$ENABLE_ANNC" -eq 1 ]; then
        if ! wait_for_serving "$INSTANCE_NUMS" $BASE_REST_PORT 90; then
            echo -e "${RED}ANNC serving failed to start${NC}"
            stop_serving
            return 1
        fi
    else
        if ! wait_for_serving "$INSTANCE_NUMS" $BASE_REST_PORT 60; then
            echo -e "${RED}Baseline serving failed to start${NC}"
            stop_serving
            return 1
        fi
    fi

    IFS='/' read -ra batch_list <<< "$batchs"
    for batch in "${batch_list[@]}"; do
        client_pids=()
        RETRIES=0

        if [ "${WARMUP_REQUESTS:-0}" -gt 0 ]; then
            warmup_client "$INSTANCE_NUMS" "$model" "$batch" "$tag"
        fi

        while [ $RETRIES -lt $MAX_RETRIES ]; do
            start_client "$INSTANCE_NUMS" "$model" "$batch" "$concurrency" "$tag"
            echo "Wait all clients..." | tee -a "$LOG_FILE"

            # Capture profiler timeline while clients are running (first attempt only)
            local profiler_pid=""
            if [ "$ENABLE_PROFILER" -eq 1 ] && [ $RETRIES -eq 0 ] && command -v python3 &>/dev/null; then
                echo "  Capturing TF Serving profiler timeline (continuous)..." | tee -a "$LOG_FILE"
                local profiler_log="$LOG_DIR/profiler_${model}_${batch}_${tag}.log"
                # Pass client PIDs so profiler loop knows when to stop
                local pid_file="$LOG_DIR/profiler_pids_${model}_${batch}"
                printf '%s\n' "${client_pids[@]}" > "$pid_file"
                (
                    cd "${SCRIPT_DIR}"
                    LD_LIBRARY_PATH=${TF_PY_SITE_PACKAGES} \
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
    local rewrite_ok=0
    local rewrite_failed=0
    local annfused_count=0
    local fusion_count=0
    local kernel_log_count=0
    local kernel_file_count=0
    for log in "$LOG_DIR"/annc_serving*_${model}.log; do
        if [ -f "$log" ]; then
            local ok_count fail_count af_count f_count k_count
            ok_count=$(grep -c "graph rewrite completed successfully" "$log" 2>/dev/null || true)
            fail_count=$(grep -c "graph rewrite failed" "$log" 2>/dev/null || true)
            af_count=$(grep -c "ANNCFused" "$log" 2>/dev/null || true)
            # With ANNC_VERBOSE=1 some backends print fused kernel names like
            # fused_dnn_embedding_hash_bucket_*; count those as valid fusions too.
            f_count=$(grep -cE "fused_[a-zA-Z0-9_]+_[0-9A-Fa-f]{8}|fused_dnn_embedding_hash_bucket" "$log" 2>/dev/null || true)
            k_count=$(grep -c "annc_generated_kernel" "$log" 2>/dev/null || true)
            ok_count=${ok_count:-0}
            fail_count=${fail_count:-0}
            af_count=${af_count:-0}
            f_count=${f_count:-0}
            k_count=${k_count:-0}
            rewrite_ok=$((rewrite_ok + ok_count))
            rewrite_failed=$((rewrite_failed + fail_count))
            annfused_count=$((annfused_count + af_count))
            fusion_count=$((fusion_count + f_count))
            kernel_log_count=$((kernel_log_count + k_count))
        fi
    done

    # Also check the work directory for generated kernel .so files, because
    # non-verbose mode does not print the kernel path in the serving log.
    local work_dir="${ANNC_WORK_BASE}/${model}_0"
    if [ -d "$work_dir" ]; then
        kernel_file_count=$(find "$work_dir" -name "annc_generated_kernel.so" -type f 2>/dev/null | wc -l)
    fi
    kernel_file_count=${kernel_file_count:-0}

    if [ "$rewrite_ok" -gt 0 ] && { [ "$annfused_count" -gt 0 ] || [ "$fusion_count" -gt 0 ] || [ "$kernel_file_count" -gt 0 ]; }; then
        echo -e "  ${GREEN}PASS${NC}: ANNC graph rewrite succeeded (${rewrite_ok}), fused kernels detected (ANNCFused=${annfused_count}, other_fusions=${fusion_count}, kernel_files=${kernel_file_count})"
    elif [ "$rewrite_ok" -gt 0 ]; then
        echo -e "  ${YELLOW}PASS${NC}: ANNC graph rewrite succeeded (${rewrite_ok}), but no fused-kernel name was detected in logs/workdir"
    else
        echo -e "  ${RED}FAIL${NC}: ANNC graph rewrite did not succeed (success=${rewrite_ok}, failed=${rewrite_failed}, ANNCFused=${annfused_count}, other_fusions=${fusion_count})"
    fi

    if [ "$kernel_log_count" -gt 0 ] || [ "$kernel_file_count" -gt 0 ]; then
        echo -e "  ${GREEN}PASS${NC}: ANNC generated kernel .so detected (log_matches=${kernel_log_count}, files=${kernel_file_count})"
    else
        echo -e "  ${YELLOW}WARN${NC}: No ANNC generated kernel .so reference found"
    fi
}

check_prerequisites

IFS=',' read -ra models_array <<< "$MODELS"

# 运行模式列表：默认一次调用内顺序跑 baseline + ANNc，也可通过 RUN_BOTH=0 回到单模式。
if [ "$RUN_BOTH" -eq 1 ]; then
    MODE_LIST=("baseline" "annc")
else
    if [ "$ENABLE_ANNC" -eq 1 ]; then
        MODE_LIST=("annc")
    else
        MODE_LIST=("baseline")
    fi
fi

for mode in "${MODE_LIST[@]}"; do
    if [ "$mode" = "annc" ]; then
        ENABLE_ANNC=1
        LOG_TAG="${ANNC_BACKEND}_annc"
    else
        ENABLE_ANNC=0
        LOG_TAG="baseline"
    fi

    LOG_DIR="${BASE_LOG_DIR}/${LOG_TAG}"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/test.log"

    echo ""
    echo "=========================================="
    echo "Running mode: ${LOG_TAG}"
    echo "LOG_DIR: ${LOG_DIR}"
    echo "=========================================="

    for model in "${models_array[@]}"; do
        run_model_test "$model"
    done
done

echo ""
echo "=========================================="
echo "All tests completed."
echo "Logs: ${BASE_LOG_DIR}/"
echo "=========================================="
