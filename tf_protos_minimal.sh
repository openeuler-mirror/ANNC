#!/bin/bash
set -e

# 使用说明
USAGE="Usage: $0 [OPTIONS]

Options:
  -d, --dir DIR       Working directory (default: current directory)
  -h, --help          Show this help message

Example:
  $0                          # Use current directory
  $0 -d /path/to/workspace    # Specify working directory"

WORK_DIR="$(pwd)"

while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--dir)
      WORK_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "$USAGE"
      exit 0
      ;;
    *)
      echo "Error: Unknown option $1"
      echo "$USAGE"
      exit 1
      ;;
  esac
done

cd "${WORK_DIR}"

# TensorFlow proto 版本
TF_TAG=v2.20.0
REPO_BASE="https://raw.githubusercontent.com/tensorflow/tensorflow/${TF_TAG}"
echo "Downloading TensorFlow proto files (version ${TF_TAG})..."
mkdir -p tf_protos_minimal/tensorflow/core/framework tf_protos_minimal/tensorflow/core/protobuf

download_proto() {
  local rel="$1"  # e.g. tensorflow/core/framework/attr_value.proto
  local dst="tf_protos_minimal/${rel}"
  mkdir -p "$(dirname "${dst}")"
  curl -L --fail -o "${dst}" "${REPO_BASE}/${rel}"
}

# framework protos
download_proto tensorflow/core/framework/attr_value.proto
download_proto tensorflow/core/framework/full_type.proto
download_proto tensorflow/core/framework/function.proto
download_proto tensorflow/core/framework/graph.proto
download_proto tensorflow/core/framework/graph_debug_info.proto
download_proto tensorflow/core/framework/node_def.proto
download_proto tensorflow/core/framework/op_def.proto
download_proto tensorflow/core/framework/resource_handle.proto
download_proto tensorflow/core/framework/tensor.proto
download_proto tensorflow/core/framework/tensor_shape.proto
download_proto tensorflow/core/framework/types.proto
download_proto tensorflow/core/framework/variable.proto
download_proto tensorflow/core/framework/versions.proto

# protobuf protos
download_proto tensorflow/core/protobuf/meta_graph.proto
download_proto tensorflow/core/protobuf/saver.proto
download_proto tensorflow/core/protobuf/saved_model.proto
download_proto tensorflow/core/protobuf/saved_object_graph.proto
download_proto tensorflow/core/protobuf/struct.proto
download_proto tensorflow/core/protobuf/trackable_object_graph.proto

echo "Generating C++ code from proto files..."
cd tf_protos_minimal

# 只针对本仓库实际需要的 proto 做最小生成
OUT_DIR=gen_code
mkdir -p "${OUT_DIR}"

protoc \
  -I. \
  --cpp_out="${OUT_DIR}" \
  tensorflow/core/framework/attr_value.proto \
  tensorflow/core/framework/full_type.proto \
  tensorflow/core/framework/function.proto \
  tensorflow/core/framework/graph.proto \
  tensorflow/core/framework/graph_debug_info.proto \
  tensorflow/core/framework/node_def.proto \
  tensorflow/core/framework/op_def.proto \
  tensorflow/core/framework/resource_handle.proto \
  tensorflow/core/framework/tensor.proto \
  tensorflow/core/framework/tensor_shape.proto \
  tensorflow/core/framework/types.proto \
  tensorflow/core/framework/variable.proto \
  tensorflow/core/framework/versions.proto \
  tensorflow/core/protobuf/meta_graph.proto \
  tensorflow/core/protobuf/saver.proto \
  tensorflow/core/protobuf/saved_model.proto \
  tensorflow/core/protobuf/saved_object_graph.proto \
  tensorflow/core/protobuf/struct.proto \
  tensorflow/core/protobuf/trackable_object_graph.proto