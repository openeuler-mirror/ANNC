# AGENTS.md

This file is the root-level project guide for both human developers and AI collaborators working on ANNC.

## 1. 项目目标与边界

ANNC 是基于 MLIR 的 AI 编译工具链，面向 openEuler 操作系统与 AArch64 平台，将 TensorFlow GraphDef 编译为 AArch64 原生代码。项目覆盖前端（TensorFlow GraphDef）、中间 IR（ATIR）、后端（AArch64 kernel）、运行时插件（ANNCOptimizer / ANNCFusedOp）、Python 绑定和测试体系。

## 2. 构建前提

- **LLVM/MLIR 21.1.3** + **nlohmann/json**：CMake `FetchContent` 在 configure 时自动从 gitee 镜像下载到 `third_party/llvm/` 和 `third_party/json/`，并作为 ANNC 构建的一部分从源码编译。无需手动克隆。
  > LLVM 首次编译需要 30-60 分钟，约 50GB 磁盘空间。增量构建仅重新编译变更文件。
- **TensorFlow**：`pip3 install tensorflow`（CMake 会在 configure 时自动检测）。也可通过 `pip3 install -r requirements.txt` 安装所有 Python 依赖。
- **pybind11 + nanobind**：`pip3 install pybind11 nanobind`。`build.sh` 会在缺少时自动安装这两者；如不希望自动安装，可传入 `--no-install-deps`。
- **protobuf-devel**：`sudo yum install -y protobuf-devel`（仅在 openEuler 上需要）
- **clang**：`annc` driver 在链接阶段需要调用 clang，构建时自动检测 `CMAKE_C_COMPILER`，运行时可通过 `ANNC_CLANG` 环境变量覆盖其路径。

## 3. 构建命令

```bash
# 一键构建（推荐）— LLVM + nlohmann/json 自动从 third_party/ 编译
./build.sh --install-prefix /opt/ANNC --build-type Debug

# 手动构建
mkdir build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/opt/ANNC \
  -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
make install
```

## 4. 运行测试

```bash
python3 -m pytest tests/          # 全部测试
python3 -m pytest tests/test_asm.py -v  # 单个测试文件
```

顶层脚本：`test_matmul.py`（matmul 功能测试）、`tf_protos_minimal.sh`（TF protobuf 最小化生成）

## 5. 代码风格

- **C++ / TableGen / MLIR TD**：使用项目根目录 `.clang-format`（Google-based，列宽 80，2 空格缩进）。提交前运行：
  ```bash
  clang-format -i annc/lib/**/*.cpp annc/include/**/*.h annc/include/**/*.td
  ```
- **Python**：遵循 PEP 8，文件使用 4 空格缩进。提交前可运行：
  ```bash
  python3 -m black python/ tests/
  ```
  （若环境未安装 black，可跳过，但需保持风格一致。）
- **CMake**：使用 2 空格缩进，关键字小写，命令与参数对齐。
- **提交信息**：使用简洁的英文或中文描述，重大变更需在 PR 描述中说明。

## 6. 架构概览

ANNC 是基于 MLIR 的 AI 编译工具链，将 TensorFlow GraphDef 编译为 AArch64 原生代码。

### 编译流水线

```
TF SavedModel/GraphDef
  ├─→ tf-adaptor → JSON 描述
  └─→ annc-tf2atir → ATIR MLIR（直接路径，跳过 JSON）

ATIR MLIR
  ├─→ annc-opt（--atir-op-fusion、--atir-block-fusion）→ 融合后的 ATIR
  ├─→ annc-fusion-metadata → 提取 ANNCFused 元数据 JSON
  ├─→ annc-asm（--atir-prune-func、--atir-tiling、--atir-unroll、--convert-atir-to-affine）
  │     └─→ lowered MLIR（affine/linalg）
  ├─→ annc → .so / 可执行文件（driver：mlir-opt → mlir-translate → opt → llc → clang link）
  ├─→ annc-verify → kernel 正确性验证
  └─→ annc-converter → TF SavedModel/GraphDef（反向转换 + GraphDef 重写）

annc-tf-pipeline → 端到端编排上述所有步骤
```

### 自定义方言：ATIR（AI Tensor IR）

- 位置：`annc/lib/Dialect/Atir/`（实现）、`annc/include/Dialect/Atir/`（TableGen TD 定义）
- 核心 Op：`MatMulOp`、`AddOp`、`ReluOp`、`ConcatOp`、`CustomizeOp`、`ConstantOp`、`VariableOp`、`ForOp`、`IfOp`、`ParallelOp`、`BufferOp`、`LoadOp`
- Passes：`OpFusion`、`EltwiseFusion`、`BlockFusion`、`Tiling`、`Unroll`、`PruneFunc`、`LLMCodeGen`、`FastCodegen`、`Initialize`、`Canonicalize`、`SelectLoweringStrategy`、`Distribute`
- Interfaces：`ShapeInfer`（形状推导）、`Interpret`（解释执行，实现在 `Interfaces/Interpret/` 中）
- OpVerify：kernel 正确性验证（通过 `kpGenLibPath` / `llmGenLibPath` 指定验证库路径）

### Lowering 路径

- `Conversion/Common/` — 公共 lowering 工具（`AtirLowering`、`InputTypeConverter`）
- `Conversion/AtirToLinalg/` — ATIR → Linalg（通用路径）
- `Conversion/AtirToAffine/` — ATIR → Affine（主要路径）
- `Target/aarch64/` — AArch64 后端 pass 集合（如 `MatmulPackAffine`、`VectorReduction` 等）

### 工具矩阵

| 工具 | 功能 |
|------|------|
| `tf-adaptor` | TF SavedModel → JSON 图描述 |
| `annc-tf2atir` | TF GraphDef → ATIR MLIR（直接，无需 JSON；含 `standalone_pb_parser`） |
| `annc-opt` | ATIR 优化（算子融合等） |
| `annc-fusion-metadata` | 从融合 ATIR 提取 ANNCFused 元数据（JSON） |
| `annc-asm` | ATIR → lowered MLIR（affine/linalg） |
| `annc` | Driver：MLIR → .so / 可执行文件 |
| `annc-verify` | Kernel 正确性验证 |
| `annc-converter` | ATIR → TF SavedModel（反向）+ GraphDef 重写 |
| `annc-tf-pipeline` | 端到端编排（`--tf-graphdef-rewrite`） |

### TensorFlow 集成

- `tensorflow_addons/annc_optimizer.cc/.h` — TF Grappler 插件，自动调用 `annc-tf-pipeline` 重写图
- `tensorflow_addons/annc_fused_op.cc/.h` — `ANNCFused` 自定义 TF Op，通过 dlopen 加载 `.so` 并调用 `_mlir_ciface_` 接口
- `tensorflow_addons/annc_fused_op_register.cc` / `annc_optimizer_register.cc` — 注册

### Python 绑定

- `python/` 使用 nanobind 将 ATIR 方言暴露给 Python（C++ 源：`AtirModule.cpp`、`Attributes.cpp`、`Types.cpp`）
- `python/annc/` 提供：`builder`、`ops`、`types`、`helper`、`enums`、`dialects/atir`

### 目录速览

| 目录 | 内容 |
|------|------|
| `annc/tools/` | CLI 工具（10 个） |
| `annc/lib/Dialect/Atir/` | ATIR 方言：Op 实现、Passes、Interfaces、OpVerify |
| `annc/lib/Conversion/` | ATIR → Affine / ATIR → Linalg + 公共工具（`Common/`） |
| `annc/lib/Target/aarch64/` | AArch64 代码生成 |
| `annc/lib/Adaptor/tensorflow/` | TF 模型解析适配器 |
| `annc/lib/Builder/` | MLIR Op 构建器 |
| `annc/lib/Kernel/` | 内置 kernel（`builtin_kernels/matmul_aarch64`）+ 线程池（`threadpool/`） |
| `annc/lib/CAPI/` | C API（Dialect + Passes） |
| `annc/include/` | 头文件（TableGen 定义、Pass 声明、Kernel API） |
| `annc/include/Dialect-c/` | C API 头文件（`Dialects.h`、`Passes.h`） |
| `python/` | Python 绑定 |
| `tensorflow_addons/` | TF Grappler 插件 + ANNCFused Op |
| `tests/` | 测试 |
| `third_party/json/` | nlohmann/json（CMake FetchContent 在 configure 时自动下载，header-only） |
| `third_party/llvm/` | LLVM/MLIR 21.1.3（CMake FetchContent 在 configure 时自动下载并从源码编译） |

## 7. 代码审查与模块治理

### 变更大小与评审粒度

- 非机械性变更建议控制在 **1000 行**以内；涉及复杂逻辑（如新增 pass、修改 lowering 策略）建议控制在 **1500 行**以内。
- 超过上述范围时，应拆分为多个可独立评审的阶段提交，并在 PR 描述中说明拆分理由。
- 单次变更尽量聚焦一个目标，避免把无关重构与功能改动混在一起。

### 外部集成面扫描

修改以下外部集成面时，必须评估破坏性变更风险，并同步更新文档与测试：

- `ANNCOptimizer` 的 `parameter_map` 接口与调用约定；
- `ANNCFusedOp` 的 attr / 输入签名 / 输出形状约定；
- Python `annc.builder` / `annc.ops` / `annc.types` 等公开 API；
- CLI 工具参数（`annc-opt`、`annc-asm`、`annc`、`annc-tf-pipeline` 等）；
- GraphDef / SavedModel 反向转换格式；
- 配置项、环境变量（如 `ANNC_CLANG`、`ANNC_BACKEND`）。

### 模块大小与拆分

- 单个源文件建议控制在 **1500 行**以内；超过 **2500 行**时，应考虑拆分为新文件或新模块，特殊情况需要在对应模块的设计文档中更新。
- 对 `annc/lib/Dialect/Atir/`、`annc/lib/Conversion/`、`annc/lib/Target/aarch64/` 等易膨胀目录尤其适用。
- 拆分时应保持相关测试和文档随模块一起移动，避免不变量远离实现代码。

### 核心模块膨胀控制

- 抵制向 `annc/lib/Dialect/Atir/` 等核心目录无限添加功能。
- 引入新概念（如新 dialect、新 target、新 frontend）时，优先考虑独立目录或独立库，而非塞进现有核心模块。

### 测试策略

- 复杂 pipeline（TF → ATIR → kernel → runtime）和优化器逻辑优先使用集成测试覆盖。
- 新增或修改 MLIR pass、kernel 生成逻辑时，应补充 `annc-verify` 或 pytest 用例。
- 测试断言优先比较整个对象/输出，而非逐字段断言。
- 涉及 MLIR 输出、GraphDef 重写结果等稳定文本输出的变更，可考虑 snapshot 测试。

## 8. 禁止事项

- 禁止提交包含 `<<<<<<<`、`=======`、`>>>>>>>` 等 Git merge conflict 标记的文件。
- 禁止在根级 `AGENTS.md` 中写入子模块的实现细节；子模块详情应放到对应子目录的 `AGENTS.md`（如 `annc/AGENTS.md`、`python/AGENTS.md`）。
- 禁止直接修改 `third_party/llvm/`、`third_party/json/` 中的源码；这些依赖由 CMake `FetchContent` 管理，本地补丁应通过 `patches/` 与 CMake 流程应用。
- 禁止未经测试修改 `ANNCOptimizer` / `ANNCFusedOp` 的公共接口（如 `parameter_map`、attr / 输入签名、Python API）。
- 禁止将构建产物、临时文件或 IDE 配置提交到仓库；具体忽略范围以 `.gitignore` 为准。

## 9. 常见陷阱

- **LLVM 首次编译**：耗时 30-60 分钟，约需 50GB 磁盘空间。若空间不足，可清理 `build/` 后使用 `ninja -j4` 或 `make -j4` 降低并行度。
- **TensorFlow 版本**：CMake 在 configure 时会自动检测当前 Python 环境下的 TF。若切换 Python 环境或 TF 版本，需重新运行 CMake。
- **`annc` driver 链接失败**：构建时自动检测 `CMAKE_C_COMPILER`，运行时可通过 `ANNC_CLANG` 环境变量指定 clang 路径。
- **Python 绑定导入失败**：确认已安装 `pybind11` 和 `nanobind`，且构建时使用了正确的 Python 解释器（`PYTHON_EXECUTABLE`）。

## 10. 文档同步规则

- 影响整体架构的变更（如新增 lowering 路径、新增后端 target、修改编译流水线）必须同步更新 `docs/annc-architecture-design.md`。
- 重大设计决策必须同步记录到 `docs/ADR/decisions.md`，并在 `docs/ADR/README.md` 索引中登记。
- CLI 工具参数变更必须同步更新源码中的 `--help` help string；若参数被写入使用文档或 README，也需同步。
- Python API（`annc.builder` / `annc.ops` / `annc.types` 等）变更必须同步更新对应示例与文档。
- `docs/api/` 为 CLI / Python API 参考文档预留位置，未来填充时应保持与代码一致。
- `docs/superpowers/` 仅用于内部治理文档（spec/plan），该目录已纳入 `.gitignore`。

## 11. 安全红线

- 不运行来源不明或未签名的第三方 `.so` / 可执行文件。
- 加载用户提供的 TensorFlow 模型或自定义 op 库前，应进行格式校验与来源确认。
- `ANNCOptimizer` 通过 `fork/exec` 调用 `annc-tf-pipeline` 时，应限定子进程可读取/写入的目录，避免访问无关系统路径。
- 编译器 pipeline 中涉及自定义 op、外部库链接、网络访问的操作，需显式评估安全影响。
