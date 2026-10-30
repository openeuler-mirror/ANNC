# ANNC

**ANNC（Accelerated Neural Network Compiler）** 是基于 MLIR 的 AI 推理编译器：把 TensorFlow GraphDef 与 ONNX 模型编译为 AArch64 原生 kernel，再回接到原框架执行。

- **统一中间表示 ATIR**：把 CPU 微架构优化（算子融合、多级 tiling、数据打包、向量化）从推理框架中解耦，一次优化、多框架复用。
- **AOT 原生 kernel**：编译期产出 `.so`；TensorFlow GEMM 融合另支持运行时 JIT 特化。
- **回接成本因框架而异**：TensorFlow 由 Grappler 插件在优化期自动重写图，业务代码无需改动；ONNX Runtime 需注册 EP 插件并把它的执行设备加入会话选项（一行注册代码，见下）。

## 项目简介

ANNC 面向 CPU 推理场景，重点解决搜索/推荐/广告（搜推广）模型中 Embedding 层与算子融合优化无法跨框架复用的问题。它把优化能力下沉到与框架无关的统一 IR，再由各框架前端/插件负责「取图—回接」：

| 维度   | 说明                                                                             |
| ---- | ------------------------------------------------------------------------------ |
| 前端   | `annc-tf2atir`（TensorFlow GraphDef）、`annc-onnx2atir`（ONNX）                     |
| 中间表示 | ATIR（AI Tensor IR），承载 shape 推导、算子融合与 kernel 选择                                 |
| 后端   | AArch64 GEMM 规划/分块/微内核 lowering，产出原生 kernel                                    |
| 回接   | TensorFlow Grappler 插件 `ANNCOptimizer` + 自定义 Op `ANNCFused`；ONNX Runtime EP 插件 |

## 支持范围与接入前提

| 维度              | 现状                                                                                               |
| --------------- | ------------------------------------------------------------------------------------------------ |
| 平台              | openEuler-24.03-LTS-SP4（系统 `llvm-toolset-20`）/ openEuler-22.03-LTS-SP4（自建 LLVM/MLIR 20/21）+ 鲲鹏 920X（AArch64）；其他平台组合未声明支持 |
| 框架              | TensorFlow（GraphDef / SavedModel）；ONNX Runtime 1.26.0                                            |
| 加速范围            | 命中 ANNC fusion pattern 的子图；未命中区域原样保留，继续由原框架 / CPU EP 执行                                          |
| TensorFlow 接入   | 会话启用 Grappler 插件 `ANNCOptimizer` 并加载 `ANNCFused` Op（均由本仓库构建）；模型与业务代码无需改动                         |
| ONNX Runtime 接入 | 注册 `libannc_ep.so` 并把其执行设备加入 `SessionOptions`（见[注册示例](addons/onnxruntime/README.md#运行)）；模型文件无需改动 |

判定依据与验收范围见 [ONNX 支持整体方案](docs/design/onnx-support/overall.md) 与 [addons/onnxruntime/README.md](addons/onnxruntime/README.md)。

## 快速开始

环境要求与依赖安装见[安装 ANNC](docs/getting-started/installation.md)。

```shell
# 1. 获取源码
git clone https://gitcode.com/openeuler/ANNC.git --branch dev
cd ANNC

# 2. 安装依赖（详见「安装 ANNC」）
pip3 install -r requirements.txt
sudo yum install -y cmake ninja-build gcc gcc-c++ python3-devel gtest-devel unzip rpm cpio
sudo yum install -y protobuf-devel protobuf-compiler
#    openEuler 24.03 使用系统 llvm-toolset-20；22.03 无该 toolset，需自建 LLVM/MLIR 20/21（见「安装 ANNC」步骤 4）
sudo yum install -y llvm-toolset-20-llvm llvm-toolset-20-llvm-devel \
  llvm-toolset-20-llvm-libs llvm-toolset-20-llvm-static \
  llvm-toolset-20-llvm-mlir llvm-toolset-20-llvm-mlir-devel \
  llvm-toolset-20-llvm-mlir-static llvm-toolset-20-python3-mlir \
  llvm-toolset-20-clang

# 3. 一键构建（LLVM/MLIR 使用系统 llvm-toolset-20 或 22.03 自建工具链；nlohmann/json 由 CMake FetchContent 获取）
#    安装到无写权限前缀时请改用有权限的目录，或确保当前用户可写
./build.sh --install-prefix "$HOME/ANNC" --build-type Debug

# 4. 验证
export PATH="$HOME/ANNC/bin:$PATH"
which annc-opt annc-asm
python3 -m pytest tests/
```

全部构建选项、CMake 选项、C++ 单元测试与覆盖率流程见[从源码构建](docs/guides/build-from-source.md)。

## 从哪里开始

| 你是              | 入口                                                                                                                                                                                                        |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 第一次接触 ANNC      | [安装 ANNC](docs/getting-started/installation.md) → [从源码构建](docs/guides/build-from-source.md) → [CLI 工作流](docs/guides/cli-workflows.md)                                                                     |
| TensorFlow 用户   | [端到端 GraphDef 重写](docs/guides/cli-workflows.md#端到端-graphdef-重写) → [运行时接入（Grappler / ANNCFused）](docs/architecture/runtime-integration.md)                                                                 |
| ONNX Runtime 用户 | [addons/onnxruntime/README.md](addons/onnxruntime/README.md)：`--enable-ort`（别名 `--enable-ort-ep`）构建 → 注册 EP → 端到端验证（真实示例 `tests/integration/ort_ep/ftdmm_context_e2e.py`）→ [EP 接入设计](docs/design/onnx-support/ep-design.md) |
| 编译器开发者          | [AGENTS.md](AGENTS.md)（工作原则与导航）→ [总体架构](docs/architecture/overview.md) → [特性与模块设计](docs/design/README.md) → [文档索引](docs/README.md)                                                                     |

完整文档索引（按分区组织）见 [docs/README.md](docs/README.md)。

## 架构速览

```text
TF SavedModel/GraphDef
  └─→ annc-tf2atir → ATIR MLIR（直接路径，跳过 JSON）

ONNX .onnx
  └─→ annc-onnx2atir → ATIR MLIR（同一 builder 引擎，ONNX 专表）

ATIR MLIR
  ├─→ annc-opt（--atir-rank-inference、--atir-identity-canonicalize、--atir-op-fusion；epilogue 融合仅在 ANNC_GEMM=2 时启用）→ 融合后的 ATIR
  ├─→ annc-fusion-metadata → 提取 ANNCFused 元数据 JSON
  ├─→ annc-asm（--atir-fast-codegen、--annc-aarch64-gemm-pipeline）
  │     └─→ lowered MLIR（linalg/affine/scf）
  ├─→ annc → .so / 可执行文件（driver：mlir-opt → mlir-translate → opt → llc → clang link）
  ├─→ annc-verify → kernel 正确性验证
  └─→ annc-converter → TF GraphDef（基于融合元数据重写为 ANNCFused 图）

annc-tf-pipeline → 端到端编排上述所有步骤

annc-onnx-pipeline → ONNX 端到端：annc-onnx2atir → annc-opt → 站点报告 → annc-asm → annc
                     （bytecode 中间件；只把 AOT 站点编译进 kernel .so）
```

- 自定义方言 ATIR 位于 `annc/lib/Dialect/Atir/`，提供 `ShapeInfer`（形状推导）与 `Interpret`（解释执行）接口，以及 OpVerify kernel 正确性验证；Op / Pass 定义在 `annc/include/Dialect/Atir/`。
- Lowering 路径：`annc/lib/Conversion/Common/`（公共 lowering 工具）、`annc/lib/Conversion/AtirToLinalg/`（ATIR → Linalg）、`annc/lib/Target/aarch64/`（AArch64 GEMM 规划、分块与微内核 lowering）。
- 各模块职责与目录导航见 [总体架构设计](docs/architecture/overview.md)。

## 工具矩阵

两条框架接入入口：**`annc-tf-pipeline`**（TensorFlow）与 **`annc-onnx-pipeline`**（ONNX Runtime）。

| 工具                         | 功能                                                                                          |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| `annc-tf-pipeline`         | TensorFlow 端到端：GraphDef → 融合 → AOT kernel `.so` → 重写后的 GraphDef                             |
| `annc-onnx-pipeline`       | ONNX 端到端：模型 → AOT kernel `.so` + 站点报告 `sites.json`，供 ONNX Runtime EP 调用                     |
| `annc-tf2atir`             | TF GraphDef → ATIR MLIR（直接转换，无需 JSON 中间步骤）                                                  |
| `annc-onnx2atir`           | [ONNX 模型 → ATIR MLIR](docs/design/onnx-support/frontend-design.md)（复用 builder 引擎，op 表按框架分离） |
| `annc-opt`                 | ATIR 优化（identity 规范化、算子融合、AOT 筛选等）                                                          |
| `annc-fusion-metadata`     | 从融合 ATIR 提取 ANNCFused 元数据（JSON 格式）                                                          |
| `annc-asm`                 | ATIR → lowered MLIR（linalg/affine/scf）                                                      |
| `annc`                     | 编译驱动：lowered MLIR → `.so` / 可执行文件                                                           |
| `annc-model-kernel-export` | 导出 lowered MLIR 中生成的 AArch64 微内核（`.S` + `manifest.json`），由 `annc` 在链接前自动调用                 |
| `annc-verify`              | Kernel 正确性验证                                                                                |
| `annc-converter`           | 基于融合元数据重写 TF GraphDef（把融合子图替换为 `ANNCFused` 节点）                                              |

各工具的典型用法与分步编译见[CLI 工作流](docs/guides/cli-workflows.md)。

## 相关项目

- [LLVM](https://llvm.org/)
- [MLIR](https://mlir.llvm.org/)
- [openEuler](https://openeuler.org/zh/)
