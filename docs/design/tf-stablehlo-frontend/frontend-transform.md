# TF GraphDef → StableHLO 前端（pb2stablehlo，component HLD）

- 状态：增量迁移中（批 1 已合入：common 层与构建骨架；工具可执行待批 8，批次计划见下）
- 源码基线：`addons/tensorflow/transform/`（1030 迁移线同构路径）
- 触发条件：仅 `ANNC_ENABLE_TF2ATIR_V2=ON` 且目标 TensorFlow 为 2.20.x 时编译（TF-2.20 分支门禁），默认构建零影响
- 决策依据：[ADR-023](../../decisions/decisions.md#adr-023)（wheel 隔离构建域、llvm 头三级来源、V2 统一开关）

## 链路位置

V2 路线的**生产端**：把 TF 模型转成 [sink 层](sink-layer.md) 消费的前端产物。

```
GraphDef / SavedModel
  └─ [pb2stablehlo]  ← 本组件
       ├─ GraphDef 导入（functionalize 控制流、shape/dtype 推导、权重外提）
       ├─ TF odml legalize → stablehlo 化
       ├─ 残差封装：不可表达/未映射能力 op → annc.residue / stablehlo.custom_call
       └─ .mlir 产物（stablehlo 主图 + 残差载体 + shape/arith 脚手架）
            └─ [residue-sink] → ATIR 混合 IR → V2 fusion / 下游 lowering
```

产物契约（与 sink 的对齐点）：残差载体为 `annc.residue`（未注册 +
`call_target_name`，与名字无关的 F2 识别形态）；wrap 边界走
`stablehlo.custom_call`（`call_target_name` = `annc.getDim`/`annc.setDim`/
`annc.encodeString`/`annc.decodeString`，属性通道保留 `residue_reason`/
`custom_op`/`custom_attrs`）。形态细节与两形态定义见
[sink-layer.md](sink-layer.md)「残差形态」。

## 推导层职责矩阵（T2）

三层推导，重叠区是**有意的分层兜底**而非冗余：

| 层 | 域 | 覆盖对象 | 推断内容 | 开关 |
| --- | --- | --- | --- | --- |
| `pb2_infer` | pb | 全部节点 | dtype | 无（必经，fail-closed） |
| `pb2_official_infer` | pb | 注册 op（改写后图） | shape/rank（`_output_shapes`） | `ANNC_NO_OFFICIAL_INFER` |
| `pb2_tf_rank_infer` | MLIR | 注册 `tf.*` + **残差载体** | rank only（规则引擎） | `ANNC_NO_TF_RANK_INFER` |

- **残差载体是 rank 引擎的第一价值面**：official 层只看注册 op，`tf.residue`
  载体里的 KP/TF op 对它不可见——只有规则引擎能给它们 rank；
- **注册 op 是互补关系**：official 成功时规则引擎对注册 op 空转无害；official
  被关闭/失败（best-effort）时规则引擎兜底（组合语义有测试：
  `TestTwoHopWithoutOfficialInference`，多跳链在无 official 世界仍传 rank）；
- `GraphPassManager`（pb 域编排框架）本批仅测试消费，T3/T5 接线。

### 与 V1 RankInference 的 parity（审计结论）

规则引擎语义移植自 V1 `annc/lib/Dialect/Atir/Passes/RankInference.cpp`
（**annc_1030 @ c754092**，ATIR 域 484 行）。逐项审计结论：

- **机制一致**：propose/commit 定点、只升 unranked、冲突永久冻结、
  StridedSlice 双向 rank 方程、MatMul rank-2 硬约束、Unique/SparseSegment/
  Shape/Rank/Size 输出约束、同形类 unify；
- **三处有意差异**：①V1 equality 类复制**实际维**（ATIR 同形契约保证维同），
  tf 域只推 rank（无维相同保证）；②V1 附带 dtype 反推，tf 域 dtype 归
  `pb2_infer`（分层）；③防御迭代上限 V1=1000、本实现=100；
- V1 侧改规则时以此清单为对照（迁移期双实现，长期以本实现为准）。

## 双 MLIR 世界隔离（核心架构约束）

本组件**不与项目 LLVM/MLIR 20 工具链共享任何头文件或库**：

| 侧 | 依赖 | 说明 |
| --- | --- | --- |
| 项目工具链域 | 系统 LLVM/MLIR 20.1.8 | sink/ATIR/lowering 等全部现有组件 |
| 本组件（wheel 域） | pip TF 2.20 wheel：`libtensorflow_cc.so.2` + wheel 内 vendored MLIR（21.0.0git） | 管线调用 TF 自身 C++ transform passes（`odml::AddLegalizeTFToStablehloPasses` 等），MLIR C++ ABI 必须与 `.so` 严格一致 |

由此派生的工程约束（均有 1030 实测记录）：

- **llvm 头三级来源**（不入库）：`third_party/llvm-include/` 本地预置快照 →
  `ANNC_TF_TRANSFORM_LLVM_INCLUDE_DIR` 显式目录 → 官方仓 sparse clone
  （`ANNC_TF_TRANSFORM_LLVM_REPO`/`_REF`，默认 `release/21.x`）；生成的
  `llvm/Config/llvm-config.h` 以构建资产形式随仓库
  （`transform/cmake/wheel-llvm-config/`）。
- **include 顺序 load-bearing**：target 级 `BEFORE` 必须压过项目全局
  `include_directories()`，否则 `llvm/` 静默解析到项目 LLVM 20
  （`ninja -t deps` 实测）。
- **属性构造不走 typed convenience 接口**：wheel 的 stablehlo 非模板构造器不在
  `libtensorflow_cc.so.2` 导出表内，op 通过 `OperationState` +
  `mlir::parseAttribute` 发射，属性拼写锚定 odml 产物形态。
- **NDEBUG 必须开启**：absl 模板跨 `.so` 边界否则 ODR 分裂。
- **测试域隔离**：`tests/unit/transform/` 用纯 CHECK 宏，不引入项目 GTest
  （GTest 属另一 MLIR 世界）。

## 批次计划（增量迁移，自 1030 `addons/tensorflow/transform`）

| 批 | 内容 | 行数量级 | 状态 |
| --- | --- | --- | --- |
| T1 | common 层（节点索引/输入引用/OpDef 加载/dtype 推导/attr 与 JSON 通道）+ 构建骨架 + 单测 | ~1070 | 已合入 |
| T2 | shape 推导层（graph pass / infer / official infer / tf rank infer） | ~840 | 已合入 |
| T3 | GraphDef 导入（pb 改写/functionalize/官方导入/编排）+ pass 声明头 | ~875 | 已合入（pb 域单测随 T8 链接闭合补齐） |
| T4 | MLIR pass 集（12 个 pass：ref 归一/executor 扁平化/源恢复/形状注记/残差封装与改名等） | ~1570 | 已合入 |
| T5 | restore + emit（残差恢复/权重规划/attr 双通道；RunPipelines 拆独立 TU 随 T8 闭合） | ~730 | 本批 |
| T6 | wrap 层（shape 组：unrank/dyn view/strided slice） | ~950 | 待迁 |
| T7 | wrap 层（string 组：transport/unwrap） | ~940 | 待迁 |
| T8 | 管线 + 入口可执行 + 集成 pytest + fixtures + CLI 文档 | ~480+测试 | 待迁 |

批次纪律：各批可独立构建；开发与测试分离提交；`ANNC_ENABLE_TF2ATIR_V2`
单一开关贯穿（与 converter-v2 / sink 侧一致）。

**T3 无单测的原因（决策记录，2026-11-11）**：

1. `pb2_import.cpp` 是单一 TU，引用 T4 的 15 个 pass 工厂与 T5 的
   `PlanWeights`/`RunPipelines` 实现——测试二进制必须链接该 TU 的全部
   符号，而这些实现尚未迁入；
2. wheel 域 OBJECT 库无链接检查，把 import 并入测试所链的
   `pb2_transform_objs` 会使全部现有测试链接失败（实测复现过）——因此
   import 独立为 `pb2_import_objs`，本批验证到编译层（0 error）；
3. 替代方案（为 T4/T5 符号造 test-only 链接替身，提前测 pb 域改写）
   评估后不采用：替身与真实实现的签名/行为漂移无防护，寿命只有两批。

补测计划：链接闭合时（T5 勘定：RunPipelines 依赖
shlo_pipeline::AddTFToStablehloPasses，其实现 tf2shlo_pipeline.cc 与
wrap 层 pass 随 T8 落地），pb 域改写（custom-op stand-in、读链重接、
argified→feed、dead-Switch 级联、feeds 装配顺序）随 T8 批补入
`tests/unit/transform/`。

**T4 测试覆盖现状（2026-11-11）**：单测覆盖 5/12——suffix strip、
executor 扁平化（正例：island 提升 + RAUW + graph/island/fetch 消失；
负例：island 数据结果数与 yield 转发数不符时整图原样跳过——该形态
经 ODS verifier 的文本路径构造不出来，负例用 C++ 手工构造，模拟
改写路径可产生的 verifier 外中间态）、registered-residue wrap（属性
搬运 + reason 区分）、carrier rename。欠账：`NormalizeRefTypes`、
`ConvergeSignatures`、`AnnotateShapesFromPb`、`NormalizeShapeVectors`
（compare/Fill 回退）、`RestoreConstSourceLoc` 溯源五件与 `PortTag`
无用例——其中溯源五件与 PortTag 依赖 pb loc 链证据，构造成本高，
随 T8 集成 pytest 的真实产物断言一并覆盖（fixtures 带 pb 原图）。

## 构建与验证

```bash
cmake -S . -B build -DANNC_ENABLE_TF2ATIR_V2=ON   # 需 TF 2.20 pip wheel
cmake --build build --target pb2_transform_objs    # 增量对象库
ctest --test-dir build -R Pb2CommonTest            # 批 1 起的单测
```

配置期门禁：TF 版本非 2.20.x、wheel 缺 mlir 头或 `libtensorflow_cc.so.2`、
llvm 头三级来源均不可用 → 明确 FATAL（含修复指引）。

## 已知差距

- 工具可执行与端到端 pytest 在批 8 落位前不可用；本文件状态行随批次推进更新。
- 1030 侧 `version.script`/rpath 链接配置在批 8 组装可执行时启用（对象库阶段
  已通过 `pb2_wheel_env` 接口库预置，单测先行验证了链接路径）。
