#!/bin/bash
set -e

USAGE="Usage: $0 [OPTIONS]

Options:
  -d, --dir DIR       Repository root (default: current directory, must contain
                      frontends/onnx/protos/onnx/onnx.proto)
  -h, --help          Show this help message

Requirements:
  - protoc (Protocol Buffers compiler) that matches the system protobuf C++
    library, either on PATH or given by PROTOC_BIN. The generated .pb.cc/.pb.h
    code will match that protoc version. This script does not depend on the
    TensorFlow proto script having run first.

The ONNX schema is vendored in-repo (frontends/onnx/protos/onnx/onnx.proto), so
unlike tf_protos_minimal.sh this script needs no network access."

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
      echo "Error: Unknown option $1" >&2
      echo "$USAGE" >&2
      exit 1
      ;;
  esac
done

cd "${WORK_DIR}"

PROTO_ROOT="frontends/onnx/protos"
PROTO_REL="onnx/onnx.proto"

if [ ! -f "${PROTO_ROOT}/${PROTO_REL}" ]; then
  echo "Error: ${PROTO_ROOT}/${PROTO_REL} not found under ${WORK_DIR}; -d must" >&2
  echo "       point at the repository root (the directory containing frontends/)." >&2
  exit 1
fi

PROTOC_BIN="${PROTOC_BIN:-}"
if [ -z "${PROTOC_BIN}" ]; then
  PROTOC_BIN="$(command -v protoc 2>/dev/null || true)"
fi
if [ -z "${PROTOC_BIN}" ] || [ ! -x "${PROTOC_BIN}" ]; then
  echo "Error: protoc not found. Install a protobuf compiler compatible with" >&2
  echo "       the system protobuf-devel package, or set PROTOC_BIN." >&2
  exit 1
fi
echo "Using protoc: ${PROTOC_BIN} ($("${PROTOC_BIN}" --version))"

cd "${PROTO_ROOT}"

OUT_DIR=gen_code
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

"${PROTOC_BIN}" \
  -I. \
  --cpp_out="${OUT_DIR}" \
  "${PROTO_REL}"

# build.sh stamps the generating protoc version so that a protoc upgrade
# triggers regeneration instead of silently reusing incompatible sources.
"${PROTOC_BIN}" --version > "${OUT_DIR}/.protoc-version"

echo "ONNX protobuf sources generated in ${PROTO_ROOT}/${OUT_DIR}"
