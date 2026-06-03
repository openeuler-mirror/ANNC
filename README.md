# ANNC-Next

## 项目简介

ANNC-Next 是基于 MLIR 的 AI 编译工具链，面向 openEuler 操作系统，将 TensorFlow GraphDef 编译为 AArch64 原生代码。项目利用 MLIR 多级中间表示的优势，实现从高层 AI 算子到底层机器码的完整编译流程。

## 环境要求

### 硬件与系统

| 项目 | 要求 |
| --- | --- |
| 操作系统 | openEuler-24.03-LTS-SP4 |
| 处理器 | 鲲鹏 920X |
| LLVM 版本 | 21.1.3 |

### 磁盘空间说明

- 完整编译（含 LLVM 从源码编译）需要约 50GB 磁盘空间
- LLVM 源码从 Gitee 镜像下载
- 首次编译 LLVM 需要 30-60 分钟，增量构建仅重新编译变更文件

### 软件依赖安装

```shell
# Python 依赖
pip install pybind11 nanobind

# TensorFlow 2.15（CMake 会在 configure 时自动检测）
pip install tensorflow==2.15.0

# 系统工具
yum install cmake clang
```

> **注意**：部分命令可能需要 `sudo` 权限执行

### LLVM & MLIR

LLVM/MLIR 21.1.3 由 CMake `FetchContent` 在 configure 时自动从 Gitee 镜像下载到 `third_party/llvm/`，并作为 ANNC 构建的一部分从源码编译，无需手动克隆或预编译。

> **注意**：LLVM 首次编译需要 30-60 分钟，约 50GB 磁盘空间。增量构建仅重新编译变更文件。

## 构建 ANNC

### 方式一：一键构建（推荐）

LLVM 从 `third_party/llvm/` 自动编译：

```shell
./build.sh --install-prefix /opt/ANNC --build-type Debug
```

> **build.sh 常用选项：**
> - `--build-type [Debug|Release|RelWithDebInfo]`：设置构建类型（默认 Release）
> - `--install-prefix <path>`：设置安装路径（默认 ./install）
> - `--enable-libcxx`：启用 libc++
> - `--clean`：清理构建目录后重新构建

### 方式二：手动构建

```shell
# 克隆仓库
git clone https://gitcode.com/li-yancheng/ANNC.git --branch dev

# 进入目录并初始化子模块
cd ANNC
git submodule init
git submodule update

# 创建构建目录
mkdir -p build && cd build

# CMake 配置
cmake -G Ninja .. \
  -DCMAKE_INSTALL_PREFIX=/opt/ANNC \
  -DCMAKE_BUILD_TYPE=Debug

# 编译并安装
ninja && ninja install
```

## 验证安装

```bash
# 设置环境变量
export PATH=/opt/ANNC/bin:$PATH

# 检查工具是否可用
which annc-opt annc-asm

# 运行测试
python3 -m pytest tests/          # 全部测试
python3 -m pytest tests/test_asm.py -v  # 单个测试文件
```

## 架构概览

### 编译流水线

```
TF SavedModel/GraphDef
  ├─→ tf-adaptor → JSON 图描述
  └─→ annc-tf2atir → ATIR MLIR（直接路径，跳过 JSON）

ATIR MLIR
  ├─→ annc-opt (--atir-op-fusion, --atir-block-fusion) → 融合后的 ATIR
  ├─→ annc-fusion-metadata → 提取 ANNCFused 元数据 JSON
  ├─→ annc-asm (--atir-prune-func, --atir-tiling, --atir-unroll, --convert-atir-to-affine)
  │     └─→ lowered MLIR (affine/linalg)
  ├─→ annc → .so / 可执行文件（driver: mlir-opt → mlir-translate → opt → llc → clang link）
  ├─→ annc-verify → kernel 正确性验证
  └─→ annc-converter → TF SavedModel/GraphDef（反向转换 + GraphDef 重写）

annc-tf-pipeline → 端到端编排上述所有步骤
```

### 自定义方言：ATIR（AI Tensor IR）

ATIR 是 ANNC 的核心自定义 MLIR 方言，专为 AI 张量计算设计。

- **源码位置**：`annc/lib/Dialect/Atir/`（实现），`annc/include/Dialect/Atir/`（TableGen TD 定义）
- **核心 Op**：`MatMulOp`、`AddOp`、`ReluOp`、`ConcatOp`、`CustomizeOp`、`ConstantOp`、`VariableOp`、`ForOp`、`IfOp`、`ParallelOp`、`BufferOp`、`LoadOp`
- **优化 Pass**：`OpFusion`、`EltwiseFusion`、`BlockFusion`、`Tiling`、`Unroll`、`PruneFunc`、`LLMCodeGen`、`FastCodegen`、`Initialize`、`Canonicalize`、`SelectLoweringStrategy`、`Distribute`
- **接口**：`ShapeInfer`（形状推导）、`Interpret`（解释执行）
- **验证**：OpVerify（kernel 正确性验证，通过 `kpGenLibPath` / `llmGenLibPath` 指定验证库）

### Lowering 路径

| 路径 | 说明 |
|------|------|
| `Conversion/AtirToLinalg/` | ATIR → Linalg（通用路径） |
| `Conversion/AtirToAffine/` | ATIR → Affine（主要路径） |
| `Target/aarch64/` | AArch64 后端：`MatmulPackAffine`、`VectorReduction`、`VectorCommonParallel`、`CacheParallel`、`CacheReduction`、`AnncOneShotBufferize` |

### 工具矩阵

| 工具 | 功能 |
|------|------|
| `tf-adaptor` | TF SavedModel → JSON 图描述 |
| `annc-tf2atir` | TF GraphDef → ATIR MLIR（直接转换，无需 JSON 中间步骤） |
| `annc-opt` | ATIR 优化（算子融合、块融合等） |
| `annc-fusion-metadata` | 从融合 ATIR 提取 ANNCFused 元数据（JSON 格式） |
| `annc-asm` | ATIR → lowered MLIR（affine/linalg） |
| `annc` | 编译驱动：MLIR → .so / 可执行文件 |
| `annc-verify` | Kernel 正确性验证 |
| `annc-converter` | ATIR → TF SavedModel（反向转换）+ GraphDef 重写 |
| `annc-tf-pipeline` | 端到端编排（`--tf-graphdef-rewrite`） |

### TensorFlow 集成

- `tensorflow_addons/annc_optimizer.cc` — TF Grappler 插件，自动调用 `annc-tf-pipeline` 重写计算图
- `tensorflow_addons/annc_fused_op.cc` — `ANNCFused` 自定义 TF Op，通过 dlopen 加载 .so 调用 `_mlir_ciface_` 接口
- `tensorflow_addons/annc_fused_op_register.cc` / `annc_optimizer_register.cc` — 注册模块

### Python 绑定

- `python/` 使用 nanobind 将 ATIR 方言暴露给 Python
- `python/annc/` 提供：`builder`、`ops`、`types`、`helper`、`enums`、`dialects/atir`

### 目录结构

| 目录 | 内容 |
|------|------|
| `annc/tools/` | CLI 工具（10 个） |
| `annc/lib/Dialect/Atir/` | ATIR 方言：Op 实现、Passes、Interfaces、OpVerify |
| `annc/lib/Conversion/` | ATIR → Affine / ATIR → Linalg lowering |
| `annc/lib/Target/aarch64/` | AArch64 代码生成 |
| `annc/lib/Adaptor/tensorflow/` | TF 模型解析适配器 |
| `annc/lib/Builder/` | MLIR Op 构建器 |
| `annc/lib/Kernel/` | 内置 kernel（matmul）+ 线程池 |
| `annc/lib/CAPI/` | C API（Dialect + Passes） |
| `annc/include/` | 头文件（TableGen 定义、Pass 声明、Kernel API） |
| `python/` | Python 绑定 |
| `tensorflow_addons/` | TF Grappler 插件 + ANNCFused Op |
| `tests/` | 测试 |
| `third_party/json/` | nlohmann/json（git submodule，header-only） |
| `third_party/llvm/` | LLVM/MLIR 21.1.3（CMake FetchContent 自动下载，从源码编译） |
| `third_party/tensorflow/` | TF C++ 依赖（CMakeLists.txt，configure 时自动检测 pip TF） |

## 使用示例

```shell
# 进入构建目录
cd /path/to/ANNC/build

# 使用 annc-opt 进行 MLIR 优化（算子融合）
./bin/annc-opt ./input.mlir --atir-op-fusion -o output.bin -emit-bytecode

# 使用 annc-asm 将 ATIR MLIR lowering 为 affine MLIR
./bin/annc-asm output.bin --atir-prune-func --atir-block-fusion --atir-tiling --atir-unroll --convert-atir-to-affine -o ../tests/annc/asm.mlir

# 使用 annc 驱动编译为 .so
cd ../tests/annc
annc -t driver_dynamic.c -o 123 asm.mlir -v --save-temps --shared

# 使用 annc-verify 验证 kernel 正确性
cd ../../build
./bin/annc-verify output.bin --atir-op-verify="kpGenLibPath=../tests/annc/fused_matmul_add_relu_A0732AD9DB33D09F.so"

# 使用 LLM CodeGen 生成 LLM 专用 TensorFlow Op Kernel
./bin/annc-asm ./output.bin --atir-LLM-CodeGen

# 使用 annc-verify 验证 LLM 生成的 kernel
./bin/annc-verify output.bin --atir-op-verify="llmGenLibPath=../annc/lib/Dialect/Atir/Passes/outputs/so/fused_matmul_add_relu_A0732AD9DB33D09F.so"
```

## 常见问题

1. **编译失败**：请确认已安装所有依赖（pybind11、nanobind、tensorflow、cmake）
2. **编译内存不足**：减少并行编译任务数，使用 `ninja -j4` 或 `make -j4` 等方式限制
3. **LLVM 下载缓慢**：Gitee 镜像偶尔不稳定，可重试或手动下载 LLVM 源码到 `third_party/llvm/`

## 相关项目

- [LLVM](https://llvm.org/)
- [MLIR](https://mlir.llvm.org/)
- [openEuler](https://openeuler.org/zh/)
