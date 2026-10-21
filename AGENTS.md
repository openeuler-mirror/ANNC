# AGENTS.md

ANNC 的根级工作约定，面向人类开发者与 AI 协作者。只保留「会改变你行动」的约束与导航；构建步骤、架构描述、工具清单在 `docs/` 维护，此处不复制。

## 1 项目范围

ANNC（Accelerated Neural Network Compiler）是基于 MLIR 的 AI 推理编译器：把 TensorFlow GraphDef 与 ONNX 模型编译为 AArch64 原生 kernel，再回接到原框架执行。

```text
TF GraphDef / SavedModel ─┐
                          ├─→ 共享 ATIR → AArch64 kernel → 框架回接
ONNX .onnx ───────────────┘
```

- 前端按框架分离（`annc-tf2atir`、`annc-onnx2atir`）；公共优化、公共 lowering 与 AArch64 后端共用。
- 回接：TF 走 Grappler 插件 `ANNCOptimizer` + `ANNCFused` Op；ONNX Runtime 走 EP 插件 `libannc_ep.so`。
- 支持平台 openEuler + AArch64（鲲鹏 920 系列），其他组合未声明支持；定位与验收范围见 [README.md](README.md)、[总体架构](docs/architecture/overview.md)。

## 2 工作原则

五条总纲，优先级高于本文件其余内容，适用于代码、配置、文档与测试的所有变更。

1. **先理解，再改动**——动工前读全上下文：Op / Pass 定义（`annc/include/Dialect/Atir/`）、实现（`annc/lib/Dialect/Atir/`）、相邻 lowering 路径与调用方、相关测试（`tests/FileCheck/`、`tests/unit/kernels/`）。凭名字或片段臆测语义即违规。
2. **复用优先**——新增前先检索已有能力：Builder（`annc/lib/Builder/`）、ATIR Op / Pass、`FusionMetadata/`、`Conversion/Common/` 公共 lowering、`Target/aarch64/`、`Kernel/` 扩展点、Python `annc.*`。已有能力可满足时不新增。
3. **非必须不引入**——不引入会长期增加维护与认知成本的新功能、新概念、新依赖、新抽象。先自问：现有能力能否满足？成本是否值得？
4. **不越界**——不为快速达成目标绕过既有抽象层（如在 Pass 内写 target-specific 代码、绕过 `Conversion/Common/`）；跨层改动先做设计评审。
5. **改完即验证**——按第 5 节选择并实际运行对应套件，不提交未验证的改动。

## 3 阅读导航

按任务取用，不必每次加载全部架构文档与 ADR。若目标目录存在 `AGENTS.md`，其局部规则优先。测试目录的分层、资产归属与前置条件见 [tests/README.md](tests/README.md)。

| 改动任务 | 优先阅读 | 验证重点 |
| --- | --- | --- |
| 构建、依赖、框架开关 | [从源码构建](docs/guides/build-from-source.md)、[安装](docs/getting-started/installation.md) | 对应构建组合配置成功、安装结果可用 |
| CLI / pipeline | [CLI 工作流](docs/guides/cli-workflows.md) | `--help`、参数与输入输出契约 |
| ATIR / 融合 / lowering / kernel | [总体架构](docs/architecture/overview.md)、[特性与模块设计](docs/design/README.md) | 对应 FileCheck；涉计算时加数值验证 |
| ONNX 导入 | [前端设计](docs/design/onnx-support/frontend-design.md) | 前端单测、CLI pytest、语义负例 |
| TF / ORT 运行时接入 | [运行时接入](docs/architecture/runtime-integration.md)、[EP 设计](docs/design/onnx-support/ep-design.md) | ABI、生命周期、失败边界、真实宿主执行 |
| Python API | [Python 绑定](docs/guides/python-binding.md) | 构建后的扩展、公开 API 与示例 |
| 文档 | [文档索引](docs/README.md) 与分区 README | 链接、索引、状态、基线与命令一致性 |
| 排障 | [常见问题](docs/guides/troubleshooting.md)、[环境变量](docs/guides/environment-variables.md) | 现象与日志是否对得上 |

**文档状态规则**：先看文首的**状态、源码基线**与**现状与差距**。当前行为以源码与测试为准，设计意图结合已接受决策理解，发布支持依赖验收证据；与源码冲突时先说明偏差，不得把提案（`proposed` ADR、待评审设计）当成既有保障。

## 4 契约保护

改动契约前先定位定义方与**全部**消费方，并同时评估 TF、ONNX 两条路径。

| 契约 | 定义 / 生产方 | 直接消费方 |
| --- | --- | --- |
| `NodeInfo` | [Builder 定义](annc/include/Builder/MLIROpBuilder.h)，TF / ONNX adapter 产出 | `MLIRBuilder` / `OnnxMLIRBuilder`（构建 ATIR） |
| Fusion Metadata | [公共定义](annc/include/FusionMetadata/FusionMetadata.h)，融合 Pass 产出属性，FusionMetadata 库解析与序列化 | `annc-fusion-metadata`、`annc-converter`、`annc-onnx-pipeline` |
| 站点报告（`sites.json`） | `annc-onnx-pipeline`（基于 Fusion Metadata） | ANNC EP |
| Kernel C ABI / 执行上下文 | 后端与 kernel（`annc/include/Kernel/ExecutionContext.h`） | TF 运行时、ORT EP |
| CLI 参数 | 各工具 | 脚本、插件、文档 |

TF 路径：`annc-converter` 把元数据转为 GraphDef 节点属性供 `ANNCFused` 读取，`ANNCOptimizer` 调用 pipeline 并读取重写后的 GraphDef。字段与版本细节以头文件及设计文档为准，此处不复述。

## 5 验证要求

按改动范围选套件，如实报告**环境、实际执行、通过、失败与跳过**；静态检查、编译成功与目标平台执行分别说明。

| 套件 | 命令 | 适用范围 |
| --- | --- | --- |
| FileCheck | `bash tests/FileCheck/run.sh [子目录]` | Pass / lowering / 融合的形态与反例回归 |
| pytest | `python3 -m pytest tests/` | 前端 CLI、融合 / lowering 静态 pipeline 契约 |
| CTest | `ctest --test-dir build --output-on-failure` | kernel、JIT 缓存、前端单测及按开关注册的测试 |
| TF 独立数值验证 | `python3 tests/e2e/tf/e2e_kp_embedding_patterns.py --case all` | Embedding 模式原图与重写图的输出对照 |
| ORT 宿主验证（CTest） | `ctest --test-dir build -R AnncEp --output-on-failure` | EP fixture 与真实 pipeline 的端到端执行 |

- `pytest tests/` **不等于**全栈验证：FileCheck 是独立入口，二者互不覆盖；pytest 在缺依赖或二进制时会**跳过**（如 `pytest.importorskip("onnx")`），跳过不构成通过证据，须说明原因或补齐环境。
- TF 独立脚本需 TensorFlow / NumPy、配套的 `annc-tf-pipeline`、`libannc_fused_op.so`、kernel 归档与目标平台工具链；非默认产物路径用 `--pipeline` / `--fused-op` 指定，库搜索路径见[环境变量参考](docs/guides/environment-variables.md)。
- ORT 宿主测试需 `ANNC_ENABLE_ONNXRUNTIME_EP=ON`、`BUILD_TESTING=ON` 与 ORT SDK / Python 依赖，上下文端到端用例还需 `annc-onnx-pipeline` 目标；运行前用 `ctest --test-dir build -N` 核对注册项，前提与用例区别见 [EP 最小端到端验证](addons/onnxruntime/README.md#最小端到端验证)。
- 覆盖率：`./build.sh --coverage --clean && cd build && ninja coverage`，详见[从源码构建](docs/guides/build-from-source.md)。

新增或修改 Pass、kernel 生成逻辑时补 FileCheck 或数值用例；断言优先比较整个对象 / 输出而非逐字段，稳定文本输出（MLIR、GraphDef 重写结果）可用 snapshot。

## 6 文档同步

**主动同步**：变更产生需要长期保留的事实（行为、契约、限制、验证结论、已知差距）时，必须在同一次交付中主动写入或修改对应文档，不得留给后续；现有文档无合适落点时就近补充，并同步索引或导航。下表只是常见映射，不作为需要同步的范围上限。

| 变更 | 需同步 |
| --- | --- |
| lowering 路径、后端 target、编译流水线 | [总体架构](docs/architecture/overview.md) |
| 重大设计决策 | `docs/decisions/decisions.md` 正文 + [ADR 索引](docs/decisions/README.md) 登记 |
| CLI 参数 | 源码 `--help`；已写入文档或 README 的同步更新 |
| Python API（`annc.builder` / `annc.ops` / `annc.types`） | 对应示例与 [Python 绑定](docs/guides/python-binding.md) |
| 环境变量、配置项 | [环境变量参考](docs/guides/environment-variables.md) |

日常 bugfix 与不影响架构的实现优化不加 ADR；`docs/superpowers/` 为内部治理文档，已 gitignore。

## 7 修改与交付边界

### 受管依赖与生成物

- 不改 `third_party/json/`；LLVM/MLIR 由系统工具链提供，ANNC 源码通过 `LLVMCompat.h` 兼容不同版本。补丁经 `patches/` 与 CMake 流程应用。
- 不改 protoc 生成产物（`frontends/onnx/protos/gen_code/`）；改 schema 或生成脚本后重新生成。

### 提交边界

- 不提交构建产物、临时文件与 IDE 配置（以 `.gitignore` 为准）；不提交含 `<<<<<<<`、`=======`、`>>>>>>>` 的文件。
- 单次变更聚焦一个目标；非机械性变更与单个 commit 原则上 ≤ 1000 行，超出时优先按可独立评审、独立回滚的逻辑单元拆分，无法独立拆分则说明原因。
- **不为凑行数拆成零散 commit**：拆分依据是「能否独立理解、独立回滚」而非行数——同一逻辑改动、彼此依赖或需一起回滚的应合并提交，几行到几十行的相邻修补不单开 commit。

### 风格

- C++ / TableGen / MLIR TD 用根目录 `.clang-format`：`clang-format -i <改动文件>`。
- Python 遵循 PEP 8 与 4 空格缩进（`python3 -m black python/ tests/`）；CMake 用 2 空格缩进。
- 提交信息用简洁中 / 英文描述，重大变更在 PR 描述中说明。

### 安全

- 不运行来源不明或未签名的第三方预编译 `.so` / 可执行文件；本地由可信源码构建的产物与受管依赖产物不在此限。加载用户提供的模型或自定义 op 库前校验格式与来源。
- `ANNCOptimizer` 经 `fork/exec` 调用 `annc-tf-pipeline` 时限定子进程可访问目录；新增外部库链接、网络访问须显式评估安全影响。

### 完成证据

- 报告完成时给出实际执行的命令与结果（含跳过项）；未实际运行的检查不得写成已通过。
