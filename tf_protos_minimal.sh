#!/bin/bash
set -e

REQUIRED_PROTOC_VERSION="3.21.9"

USAGE="Usage: $0 [OPTIONS]

Options:
  -d, --dir DIR       Working directory (default: current directory)
  -h, --help          Show this help message

Requirements:
  - protoc (Protocol Buffers compiler) version ${REQUIRED_PROTOC_VERSION}
  - The script will automatically download protoc ${REQUIRED_PROTOC_VERSION} if the
    system protoc version does not match
  - protoc ${REQUIRED_PROTOC_VERSION} must match the TensorFlow bundled protobuf
    version to avoid header/library version conflicts at compile time

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

resolve_protoc() {
  local protoc_bin="${PROTOC_BIN:-}"
  if [ -n "${protoc_bin}" ]; then
    echo "${protoc_bin}"
    return 0
  fi

  local system_protoc
  system_protoc="$(command -v protoc 2>/dev/null || true)"
  if [ -n "${system_protoc}" ]; then
    local version
    version=$("${system_protoc}" --version 2>/dev/null | awk '{print $2}')
    if [ "${version}" = "${REQUIRED_PROTOC_VERSION}" ]; then
      echo "${system_protoc}"
      return 0
    fi
    echo "System protoc version is ${version}, but ${REQUIRED_PROTOC_VERSION} is required (must match TensorFlow bundled protobuf)." >&2
  fi

  local local_protoc="${PWD}/.protoc-${REQUIRED_PROTOC_VERSION}/bin/protoc"
  if [ -x "${local_protoc}" ]; then
    echo "${local_protoc}"
    return 0
  fi

  echo "Downloading protoc ${REQUIRED_PROTOC_VERSION} for aarch64..." >&2
  mkdir -p "${PWD}/.protoc-${REQUIRED_PROTOC_VERSION}"
  local tmpzip
  tmpzip="$(mktemp /tmp/protoc-XXXXXX.zip)"
  curl -L -o "${tmpzip}" \
    "https://github.com/protocolbuffers/protobuf/releases/download/v21.9/protoc-21.9-linux-aarch_64.zip"
  unzip -o "${tmpzip}" -d "${PWD}/.protoc-${REQUIRED_PROTOC_VERSION}"
  rm -f "${tmpzip}"

  if [ -x "${local_protoc}" ]; then
    echo "${local_protoc}"
    return 0
  fi

  echo "Error: Failed to download protoc ${REQUIRED_PROTOC_VERSION}." >&2
  exit 1
}

PROTOC_BIN="$(resolve_protoc)"
echo "Using protoc: ${PROTOC_BIN} ($("${PROTOC_BIN}" --version))"

# TensorFlow proto 版本
TF_TAG=v2.20.0
REPO_BASE="https://gitee.com/mirrors/tensorflow/raw/${TF_TAG}"
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

"${PROTOC_BIN}" \
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