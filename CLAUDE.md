# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 构建前提

- **LLVM/MLIR 21.1.3** + **nlohmann/json**: CMake `FetchContent` 在 configure 时自动从 gitee 镜像下载到 `third_party/llvm/` 和 `third_party/json/`，并作为 ANNC 构建的一部分从源码编译。无需手动克隆。
  > LLVM 首次编译需要 30-60 分钟，约 50GB 磁盘空间。增量构建仅重新编译变更文件。
- **TensorFlow**: `pip3 install tensorflow` (CMake 会在 configure 时自动检测)
- **pybind11 + nanobind**: `pip3 install pybind11 nanobind`
- **protobuf-devel**: `sudo yum install -y protobuf-devel` (仅在 openEuler 需要)
- **clang**: `annc` driver 在链接阶段需要调用 clang，构建时自动检测 `CMAKE_C_COMPILER`，运行时可通过 `ANNC_CLANG` 环境变量覆盖

## 构建

```bash
# 一键构建 (推荐) — LLVM + nlohmann/json 自动从 third_party/ 编译
./build.sh --install-prefix /opt/ANNC --build-type Debug

# 手动构建
mkdir build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/opt/ANNC \
  -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
make install
```

## 运行测试

```bash
python3 -m pytest tests/          # 全部测试
python3 -m pytest tests/test_asm.py -v  # 单个测试文件
```

顶层脚本: `test_matmul.py` (matmul 功能测试), `tf_protos_minimal.sh` (TF protobuf 最小化生成)

## 架构概览

ANNC 是基于 MLIR 的 AI 编译工具链，将 TensorFlow GraphDef 编译为 AArch64 原生代码。

### 编译流水线

```
TF SavedModel/GraphDef
  ├─→ tf-adaptor → JSON 描述
  └─→ annc-tf2atir → ATIR MLIR (直接路径，跳过 JSON)

ATIR MLIR
  ├─→ annc-opt (--atir-op-fusion, --atir-block-fusion) → 融合后的 ATIR
  ├─→ annc-fusion-metadata → 提取 ANNCFused 元数据 JSON
  ├─→ annc-asm (--atir-prune-func, --atir-tiling, --atir-unroll, --convert-atir-to-affine)
  │     └─→ lowered MLIR (affine/linalg)
  ├─→ annc → .so / 可执行文件 (driver: mlir-opt → mlir-translate → opt → llc → clang link)
  ├─→ annc-verify → kernel 正确性验证
  └─→ annc-converter → TF SavedModel/GraphDef (反向转换 + GraphDef 重写)

annc-tf-pipeline → 端到端编排上述所有步骤
```

### 自定义方言: ATIR (AI Tensor IR)

- 位置: `annc/lib/Dialect/Atir/` (实现), `annc/include/Dialect/Atir/` (TableGen TD 定义)
- 核心 Op: `MatMulOp`, `AddOp`, `ReluOp`, `ConcatOp`, `CustomizeOp`, `ConstantOp`, `VariableOp`, `ForOp`, `IfOp`, `ParallelOp`, `BufferOp`, `LoadOp`
- Passes: `OpFusion`, `EltwiseFusion`, `BlockFusion`, `Tiling`, `Unroll`, `PruneFunc`, `LLMCodeGen`, `FastCodegen`, `Initialize`, `Canonicalize`, `SelectLoweringStrategy`, `Distribute`
- Interfaces: `ShapeInfer` (形状推导), `Interpret` (解释执行, 实现在 `Interfaces/Interpret/` 中)
- OpVerify: kernel 正确性验证 (kpGenLibPath / llmGenLibPath, 实现在 `OpVerify/` 中)
- 辅助: `Passes/Patterns/` (模式注册), `Passes/llm/` (LLM CodeGen 脚本), `Passes/scripts/` (编译/测试脚本)

### Lowering 路径

- `Conversion/Common/` — 公共 lowering 工具 (`AtirLowering`, `InputTypeConverter`)
- `Conversion/AtirToLinalg/` — ATIR → Linalg (通用路径)
- `Conversion/AtirToAffine/` — ATIR → Affine (主要路径)
- `Target/aarch64/` — AArch64 后端: `MatmulPackAffine`, `VectorReduction`, `VectorCommonParallel`, `CacheParallel`, `CacheReduction`, `AnncOneShotBufferize`, `Aarch64CodegenPipelineBuild`

### 工具矩阵

| 工具 | 功能 |
|------|------|
| `tf-adaptor` | TF SavedModel → JSON 图描述 |
| `annc-tf2atir` | TF GraphDef → ATIR MLIR (直接, 无需 JSON; 含 `standalone_pb_parser`) |
| `annc-opt` | ATIR 优化 (算子融合等) |
| `annc-fusion-metadata` | 从融合 ATIR 提取 ANNCFused 元数据 (JSON) |
| `annc-asm` | ATIR → lowered MLIR (affine/linalg) |
| `annc` | Driver: MLIR → .so / 可执行文件 |
| `annc-verify` | Kernel 正确性验证 |
| `annc-converter` | ATIR → TF SavedModel (反向) + GraphDef 重写 |
| `annc-tf-pipeline` | 端到端编排 (`--tf-graphdef-rewrite`) |

### TensorFlow 集成

- `tensorflow_addons/annc_optimizer.cc/.h` — TF Grappler 插件, 自动调用 `annc-tf-pipeline` 重写图
- `tensorflow_addons/annc_fused_op.cc/.h` — `ANNCFused` 自定义 TF Op, dlopen 加载 .so 调用 `_mlir_ciface_` 接口
- `tensorflow_addons/annc_fused_op_register.cc` / `annc_optimizer_register.cc` — 注册

### Python 绑定

- `python/` 使用 nanobind 将 ATIR 方言暴露给 Python (C++ 源: `AtirModule.cpp`, `Attributes.cpp`, `Types.cpp`)
- `python/annc/` 提供: `builder`, `ops`, `types`, `helper`, `enums`, `dialects/atir`

### 目录速览

| 目录 | 内容 |
|------|------|
| `annc/tools/` | CLI 工具 (10 个) |
| `annc/lib/Dialect/Atir/` | ATIR 方言: Op 实现、Passes、Interfaces、OpVerify |
| `annc/lib/Conversion/` | ATIR → Affine / ATIR → Linalg + 公共工具 (`Common/`) |
| `annc/lib/Target/aarch64/` | AArch64 代码生成 |
| `annc/lib/Adaptor/tensorflow/` | TF 模型解析适配器 |
| `annc/lib/Builder/` | MLIR Op 构建器 |
| `annc/lib/Kernel/` | 内置 kernel (`builtin_kernels/matmul_aarch64`) + 线程池 (`threadpool/`) |
| `annc/lib/CAPI/` | C API (Dialect + Passes) |
| `annc/include/` | 头文件 (TableGen 定义, Pass 声明, Kernel API) |
| `annc/include/Dialect-c/` | C API 头文件 (`Dialects.h`, `Passes.h`) |
| `python/` | Python 绑定 |
| `tensorflow_addons/` | TF Grappler 插件 + ANNCFused Op |
| `tests/` | 测试 |
| `third_party/json/` | nlohmann/json (CMake FetchContent 在 configure 时自动下载, header-only) |
| `third_party/llvm/` | LLVM/MLIR 21.1.3 (CMake FetchContent 在 configure 时自动下载，从源码编译) |
