# TF GraphDef → StableHLO 前端（pb2stablehlo）设计

| 文档控制项 | 内容 |
| --- | --- |
| 文档编号 | ANNC-TFSTABLEHLO-LLD-001 |
| 版本与日期 | 1.0，2026-11-12 |
| 文档状态 | 模块设计。描述当前设计意图与实现遵循度，差距见第 9 节 |
| 评审对象 | ANNC 架构师、TF 前端与 sink 层开发、测试负责人 |
| 源码基线 | `addons/tensorflow/transform/`（17 个 TU，含入口 `pb2stablehlo.cpp`；源码基线 `810dd12`，分支 `feature/transform-final`） |
| 上级文档 | [总体架构](../../architecture/overview.md)（V2 stablehlo 路线一节）、[tf-stablehlo-frontend 设计组](README.md) |
| 关联文档 | [sink 层设计](sink-layer.md)（产物消费方）、[ADR-023](../../decisions/decisions.md#adr-023)（wheel 隔离构建域）、[CLI 工作流](../../guides/cli-workflows.md)、[环境变量参考](../../guides/environment-variables.md) |

## 1 概述

### 1.1 定位与作用

本组件是 V2 stablehlo 路线的**生产端**：把 TensorFlow GraphDef 转成 sink 层
消费的前端产物（stablehlo 主图 + 残差载体 + 形状/算术脚手架的 `.mlir`）。
TF 2.20 pip wheel 内置的官方 odml legalize passes 承担 tf→stablehlo 的主体
转换；本组件在其前后布置 pb 域改写、推导、残差封装与回滚，使官方路径能
处理 KP 域算子、string 载荷与不可静态推导的形状；无法静态推导的语义
一律以残差形式保留原图计算，不做默认假设、不丢弃算子。

### 1.2 设计目标与优先级

| 编号 | 目标 | 优先级 | 判据 |
| --- | --- | --- | --- |
| G1 | 语义保真，fail-closed：无法推导的形态进入残差通道，不做默认假设 | 最高 | 未注册 op、无 dtype 槽、缺属性实例一律残差或拒绝；单测含对应负例 |
| G2 | 官方路径复用最大化：tf→stablehlo 主体交给 wheel 内 odml passes | 最高 | 本组件不自建 legalize 模式；wrap 层只做形状等价的边界改写 |
| G3 | wheel 隔离：与项目 LLVM/MLIR 20 工具链零共享 | 最高 | `ninja -t deps` 无跨域头解析；ADR-023 |
| G4 | 默认构建零影响 | 高 | `ANNC_ENABLE_TF2ATIR_V2=OFF`（默认）时不配置、不编译本组件任何 TU |
| G5 | 每个非官方 pass 可独立 A/B | 中 | 6 个 `ANNC_NO_*` 开关逐一关闭后管线仍完成（残差回退；当前集成级覆盖仅 UNRANK_WRAP/STRIDED_SLICE_DYN，其余见 GAP-2） |

### 1.3 范围

在范围内：`addons/tensorflow/transform/` 全部 17 个 TU（含入口
`pb2stablehlo.cpp`）——pb 域改写与导入、
三层推导、MLIR pass 集、残差封装与恢复、wrap 层（shape/string 两组）、
管线编排、`pb2stablehlo` 入口与集成测试。

范围之外：sink 层对残差的映射消费（见
[sink-layer.md](sink-layer.md)）、TF Grappler 插件与运行时回接、ONNX 前端、
V2 融合。

### 1.4 术语

| 术语 | 含义 |
| --- | --- |
| 残差载体 | 承载本组件无法转成 stablehlo 的 TF op 的 IR 形态，sink 按 F1/F2 两形态识别（§3.3） |
| 编解码包裹 | `annc.encodeString → tf.op(i8 载荷) → annc.decodeString` 的三段包裹结构，使官方 legalize 在数值域处理 string 载荷（源码注释中称 sandwich） |
| Smuggle 护栏 | legalize 前置的崩溃防御（`SmuggleUnrankedOpsPass`）：将已实证会触发 odml 模式段错误的形态提前封装为 `stablehlo.custom_call` |
| rank 引擎 | `pb2_tf_rank_infer` 的规则引擎，对注册 op 与残差载体推 rank |
| 双 MLIR 世界 | 项目工具链（LLVM/MLIR 20.1.8）与 wheel 域（vendored MLIR 21.0.0git）并存的构建事实（§4.5） |
| 载体双名 | 载体名经统一改名：管线内 rename 之前为 `tf.residue`，产物拼写统一为 `annc.residue`（改名置于整链最后，见 S3）。匹配谓词同时接受两种拼写（所处管线位置与测试构造的 IR 拼写不同）；sink 识别的权威依据是 `call_target_name` 属性，与载体名拼写无关（同 sink-layer.md G4/C4） |

## 2 设计约束、假设与原则

### 2.1 隔离约束

| 约束 | 内容 | 违反后果 |
| --- | --- | --- |
| I1 域隔离 | 本组件仅链接 wheel 的 `libtensorflow_cc.so.2` 与其 vendored MLIR；不 include 项目 LLVM 20 头 | 头静默解析到项目域 → ABI 错位段错误（`ninja -t deps` 实测可查） |
| I2 属性构造 | 不使用 wheel 的 typed convenience 构造器（不在 `.so` 导出表内） | 链接期 undefined symbol |
| I3 NDEBUG | 跨 `.so` 边界的 absl 模板必须 NDEBUG 一致 | ODR 分裂，运行期行为不定 |
| I4 方言注册 | MLIRContext 单线程 + 允许未注册方言 + 经官方 importer dummy 图注册 TF 方言 | 手搭 registry 与 `.so` 分裂，段错误 |

### 2.2 语义约束

| 约束 | 内容 | 违反后果 |
| --- | --- | --- |
| S1 fail-closed | dtype/shape/属性无法推导 → 残差或拒绝（exit 非 0），不引入默认值 | 静默错误数值 |
| S2 形状等价改写 | wrap 层只做序列域/切片域等价的边界改写；改不等价的（广播二元、动态 Reshape 视图）禁止 | 错误数学（T6 检视实证） |
| S3 rename 时序 | 载体统一改名（`tf.residue`→`annc.residue`）必须晚于一切按名字溯源的 pass，且为全量改名 | 溯源判定失效或残留 tf 拼写（T5 检视实证，rename 位于最末 PM） |
| S4 常量不可读即放弃 | Smuggle 与维度适配器无法读取常量操作数时跳过该 op | 以安全默认值覆盖真实语义 |

### 2.3 设计原则

1. 官方路径优先：能用 odml 的不自己写；本组件的 pass 都是官方路径的
   前置（prepare）或后置（restore/unwrap），不改写官方已能消化的形态。
2. 单一开关：`ANNC_ENABLE_TF2ATIR_V2` 贯穿 sink/converter/本组件，不另设开关。
3. 双名匹配：残差载体识别与名字解耦（属性 `call_target_name` 为准），
   载体改名不改变语义。
4. 产物自描述：残差保留完整源属性与 `residue_reason`，sink 仅凭产物即可解析，无需前端上下文。

## 3 总体设计

### 3.1 管线阶段

```text
GraphDef.pb
  └─[入口 pb2stablehlo] CLI 解析 → LoadGraph（索引/槽位）→ SetupMlirContext（I4）
      └─[RunOfficialRoute]
          PM1(pb 域): pb2_import 改写（custom-op stand-in / argified→feed /
                      dead-Switch 级联 / 权重外提）
          PM2(导入+MLIR 域): 官方 Graphdef→Mlir 导入 → 12 个 MLIR pass
                      （T4：ref 归一 / executor 扁平化 / 源恢复 / 形状注记 /
                       残差封装 / 签名收敛等）
          PM3(legalize): TfRankInfer → UnrankWrap → StridedSliceDyn →
                      StringTransport → Smuggle → odml AddLegalizeTFToStablehlo
                      → StringUnwrap → SyncFuncSignature
          emit: AppendCustomOpAttrs + PlanWeights → GraphDef/MLIR 产物
```

| 阶段 | 输入 | 输出 | 失败语义 | 开关 |
| --- | --- | --- | --- | --- |
| pb 改写 | 原图 + op 表 | 改写后图（可导入） | 图结构不可恢复 → exit 6 | 无（必经） |
| 官方导入 | 改写后图 | tf MLIR（executor 域） | 导入失败 → exit 1 | `--skip-preprocess` 可跳过官方 preprocess |
| MLIR pass 集 | tf MLIR | 规范化 tf MLIR + 残差载体 | 结构非法形态整图原样跳过（保持原样，交由残差通道回退） | 无 |
| TfRankInfer | tf MLIR | rank 注记 | 无法推导时不注记（rank 引擎只升不降） | `ANNC_NO_TF_RANK_INFER` |
| UnrankWrap | unranked 一元 op | getDim/setDim 序列视图边界 | 非 allowlist / 非一元 → 不改写 | `ANNC_NO_UNRANK_WRAP` |
| StridedSliceDyn | 常量界限动态切片 | real_dynamic_slice + shrink 折叠 | 非/弱形态（动态维 shrink 等）→ 残差保留 | `ANNC_NO_STRIDED_SLICE_DYN` |
| StringTransport | string 载荷数据操作（22 个 allowlist） | 编解码包裹结构 | axis 不可读 / 缺属性 → 不编码 | `ANNC_NO_STRING_TRANSPORT` |
| Smuggle | 全图 tf op（注册且数值载荷） | 危险形态 → custom_call | 命中名单才改写；名单外不做处理 | 无（护栏） |
| odml legalize | 规范 tf MLIR | stablehlo 主图 + 残差 | 模式失败 → 残差通道 | 无 |
| StringUnwrap | legalize 后的包裹结构 | 回滚未被转换的包裹 | tf 副本已合法化 → 仅移除边界操作 | `ANNC_NO_STRING_UNWRAP` |
| 官方富化（pb 域） | 改写后图 | `_output_shapes` 注记 | 失败 best-effort，由 rank 引擎回退承担 | `ANNC_NO_OFFICIAL_INFER` |

### 3.2 产物形态

`.mlir` 文本，含：stablehlo 主图（`func.func` + `tf.entry_function`）、
残差载体（F1/F2，见 §3.3）、形状/算术脚手架（getDim/setDim、常量）。
默认写 stdout（`--no-dump-out` 关闭），`-o <file>` 改为写入文件；
`--dump` 另行打印中间 tf MLIR。

### 3.3 与 sink 的契约

| 契约 | 内容 | 消费方校验点 |
| --- | --- | --- |
| F1 形态 | `stablehlo.custom_call`，`call_target_name` = TF op 名或 wrap 边界名（`annc.getDim`/`annc.setDim`/`annc.encodeString`/`annc.decodeString`），属性通道带 `residue_reason`/`custom_op`/`custom_attrs` | sink `TfResidueProvider` 支持双形态（F1+F2），见 sink-layer.md「残差形态」 |
| F2 形态 | 未注册 op、带 `call_target_name` 声明（`annc.residue` 即此形态） | 同上 |
| op 表 | sink `kOpMap`（complement-only 域）之外的 TF op 不映射，保持残差原样 | sink 侧裁决，前端不假设映射成功 |

## 4 核心机制

### 4.1 三层推导（重叠是有意的分层回退）

| 层 | 域 | 覆盖对象 | 推断内容 | 开关 |
| --- | --- | --- | --- | --- |
| `pb2_infer` | pb | 全部节点 | dtype | 无（必经，S1） |
| `pb2_official_infer` | pb | 注册 op（改写后图） | shape/rank（`_output_shapes`） | `ANNC_NO_OFFICIAL_INFER` |
| `pb2_tf_rank_infer` | MLIR | 注册 `tf.*` + 残差载体 | rank only（规则引擎） | `ANNC_NO_TF_RANK_INFER` |

残差载体是 rank 引擎的首要适用对象：official 层只覆盖注册 op，载体内的
KP op 仅能由规则引擎推导 rank。official 成功时引擎对注册 op 无副作用；
official 关闭或失败时由规则引擎回退承担（`TestTwoHopWithoutOfficialInference`
锁定该组合语义）。
规则语义移植自 V1 `RankInference.cpp`（parity 审计：propose/commit 定点、
只升 unranked、冲突冻结、StridedSlice 双向方程、MatMul rank-2 硬约束一致；
有意差异三处：tf 域只推 rank 不复制维、dtype 归 pb 层、迭代上限 100 vs 1000）。

### 4.2 残差封装与恢复

- **pb 域**：未注册 op 以 stand-in 形态过官方导入（完整 NodeDef 存表），
  导入后 `WrapRegisteredResiduePass` 换回载体并回填属性；argified 输入重接
  feed、dead-Switch 级联清理、权重外提为常量。
- **MLIR 域**：12 个 pass（§3.1 PM2），其中 `RestoreConstSourceLocPass`
  把 pb 域 loc 链证据回填到 MLIR 常量（溯源五件）；`RenameResidueCarrierPass`
  在 PM3 执行改名（S3 时序约束）。
- **回滚**：legalize 后 `StringUnwrap` 校验包裹中的 tf 副本是否已由官方
  legalize 转换；未被转换的包裹沿数据链回溯，重新接续原 string 操作数并
  移除边界操作，恢复原 op。

### 4.3 string 编解码包裹

string 载荷对 odml 数值域不可见。承载 string 载荷的数据操作（22 个
allowlist，含
DynamicStitch/ConcatV2/GatherV2 等布局敏感算子）包成
`encodeString → tf.op(i8) → decodeString`：行格式 `[len i32][payload][pad]`；
多操作数 emitNormalizeToMax 位宽归一；常量操作数折叠打包。维度适配器按
算子定位 axis（Concat 首位、ConcatV2/GatherV2 末位 rank-0），读不到即放弃
编码（S4）。DynamicStitch 布局固定为前 N indices + 后 N data。

### 4.4 Smuggle 护栏

odml legalize 对部分形态**段错误**（非转换失败）。护栏在 legalize 前把
命中形态换入 `stablehlo.custom_call`（原名进 `call_target_name`，形状决策
移交 sink/后端）：

| 分支 | 依据 |
| --- | --- |
| 操作数或结果 unranked | tf.GatherNd with unranked params 段错误（bisect 实证） |
| tf.Mul 操作数元素类型不一致 | 同 annc-cvr 调查实证；corpus 中该形态仅现于 Mul，故不泛化 |
| Dot 家族（MatMul/BatchMatMulV2/Einsum）rank<2 操作数 | rank floor 违约实证；TF 标准注册表无 BatchMatMulV3/XlaDot 同类漏网 |

名单是**经验清单**：新崩溃形态靠实测发现后增补；其本质是针对上游 pattern 缺少
shape 前置检查这一前提，在本地以护栏补偿；长期方案在上游修复。

### 4.5 双 MLIR 世界隔离（构建域）

| 侧 | 依赖 |
| --- | --- |
| 项目工具链域 | 系统 LLVM/MLIR 20.1.8（sink/ATIR/lowering 全部现有组件） |
| 本组件（wheel 域） | TF 2.20 pip wheel：`libtensorflow_cc.so.2` + vendored MLIR 21.0.0git |

工程约束：llvm 头三级来源（本地快照 `third_party/llvm-include/` →
`ANNC_TF_TRANSFORM_LLVM_INCLUDE_DIR` → 官方仓 sparse clone，生成的
`llvm-config.h` 以构建资产入库）；target 级 `BEFORE` include 压过项目全局
路径；I1~I4 见 §2.1。配置期对 TF 版本、wheel 产物、llvm 头来源逐一
FATAL 门禁（含修复指引）。

## 5 关键设计决策

| 决策 | 理由 | 被否备选 | 影响 |
| --- | --- | --- | --- |
| dyn-view 不迁 | 动态 Reshape 短路成 getDim/setDim 会引入一维扁平中间形态，属不需要的形状结构变化；sink 侧另有 rank-refine 处理路径 | 原样迁移（1030 行为） | 动态 Reshape 走残差；集成测试相应断言已对齐 |
| 二元广播族移出 unrank allowlist | 压平错配形状是错误数学（T6 检视实证） | 保留（1030 行为） | unrank 二元由 Smuggle 护栏承接 |
| `--op_list` 不迁 | OpDef 来源统一为 wheel 注册表（决策记录），外部 OpList 引入第二来源 | 保留 CLI 兼容 | 入口无 `--op_list`；用法见 cli-workflows |
| shrink+动态维不翻译 | 运行时钳制的 start 配常量 limit 可能越界，静态不可证安全 | 运行时钳制表达式 | 该形态残差保留；集成测试 killswitch 用例以动态输入锚定 |
| T3 pb 域单测随 T8 补 | import TU 链接依赖 T4/T5 符号，OBJECT 库无链接检查，替身方案无防漂移能力 | test-only 链接替身 | `pb2_import_objs` 于 T8 并回主库 |

## 6 失败模型

| 失败类别 | 触发 | 行为 | 出口 |
| --- | --- | --- | --- |
| 配置期门禁 | TF 非 2.20.x / wheel 缺产物 / llvm 头不可用 | FATAL + 修复指引 | cmake configure 失败 |
| CLI 用法错误 | 未知选项 / 缺值 / 缺图路径 | usage + FATAL | exit 2 |
| IO/图结构 | 文件不可读 / proto 解析失败 / 重名节点 | FATAL + 原因 | exit 1 / 6 |
| 方言注册失败 | dummy 导入不成功 | FATAL | exit 3 |
| 类型/不变量 | dtype 无法推导（无 attr/OpDef/下游约束） | FATAL（`--assume-opaque-i8` 可标记哨兵） | exit 4 |
| 内部阶段 | 管线阶段返回失败 | 带阶段名报错 | exit 5 |
| pass 级跳过 | 本文档 §3.1 各 pass 的失败语义 | 跳过/残差，管线继续 | 无（诊断打印） |
| 集成测试缺前提 | pb2stablehlo 未构建 | pytest 整文件 skip | skip 不构成通过证据 |

## 7 质量属性与验证

| 层 | 资产 | 覆盖 | 命令 |
| --- | --- | --- | --- |
| 单元 | 7 套件（common/infer/tf_rank_infer/mlir_passes/t5/wrap/string），~190 断言 | 推导规则、12 pass 的正反例、wrap 边界、编解码包裹结构与回滚 | `ctest --test-dir build -R pb2_` |
| 集成 | `tests/integration/tf_transform/` 5 文件 30 用例（手工 GraphDef → 可执行 → 产物断言） | rank 恢复、切片直译与 killswitch、包裹结构端到端、unrank 边界、StringUnwrap | `python3 -m pytest tests/integration/tf_transform/` |
| FileCheck | `tests/FileCheck/Conversion/tf-residue-v2-sink.mlir` | sink 侧消费本组件产物形态 | `bash tests/FileCheck/run.sh Conversion` |

改动纪律：新增/修改 pass 必须补对应层用例；断言优先整图/整产物比较；
killswitch（`ANNC_NO_*`）改动须同时验证关闭路径的残差回退形态。

## 8 扩展点与演进

- **wrap allowlist 增补**：新 op 进 string/unrank 名单前须证明形状等价
  （S2）并补正反例；名单即契约，增补随测试同批交付。
- **Smuggle 名单增补**：新崩溃形态附 bisect 记录；不预判、不泛化。
- **新 kill-switch**：随 pass 引入，命名 `ANNC_NO_<PASS>`，登记
  [环境变量参考](../../guides/environment-variables.md)。
- **上游修复回退**：odml 修复某崩溃形态后，对应 Smuggle 分支可移除
  （以 wheel 版本门为界，不保留双路径）。

## 9 现状与目标差距

| 编号 | 差距 | 影响 | 计划 |
| --- | --- | --- | --- |
| GAP-1 | preprocess 路径零覆盖（集成测试全走 `--skip-preprocess`） | 官方 preprocess/freeze 分支的交互未锚定 | 补 1~2 个走 preprocess 的集成用例 |
| GAP-2 | 6 个 kill-switch 中仅 UNRANK_WRAP/STRIDED_SLICE_DYN 有集成级验证 | 其余开关关闭路径仅单测覆盖 | 集成用例补齐 |
| GAP-3 | T4 12 pass 中 7 个无单测（NormalizeRefTypes/ConvergeSignatures/AnnotateShapesFromPb/NormalizeShapeVectors/溯源五件/PortTag） | 构造成本高（pb loc 链证据、verifier 外中间态） | 按风险排序补；集成测试已覆盖其组合行为 |
| GAP-4 | 载体双名的 TF 侧（`tf.residue` 形态）与 sink F2 的对齐测试在 sink 侧 | 前端不锁 sink 消费语义 | 由 sink-layer.md 测试矩阵维护 |

## 附录 A 源码位置索引（文件级）

| 文件 | 职责 |
| --- | --- |
| `common/pb2_common.{h,cpp}` | 节点索引/输入引用/OpDef 加载/共享诊断 |
| `common/pb2_attrs.cpp` | attr 与 JSON 通道 |
| `graph/pb2_graph_pass.cpp` | pb 域 GraphPassManager 框架 |
| `graph/pb2_infer.cpp` | dtype 推导（fail-closed 入口） |
| `graph/pb2_official_infer.cpp` | 官方 GraphProperties 富化 |
| `graph/pb2_import.cpp` | pb 改写与官方导入编排 |
| `passes/pb2_tf_rank_infer.cpp` | rank 规则引擎 |
| `passes/pb2_mlir_passes.cpp` | 12 个 MLIR pass（工厂清单见文件尾） |
| `passes/pb2_restore_pass.cpp` | 残差恢复（RestoreCtx + 8 阶段函数） |
| `pipeline/pb2_emit.cpp` | attr 双通道 / 权重规划 |
| `pipeline/pb2_run_pipelines.cpp` | PM1~PM3 编排 + allShaped 门 + rename 时序 |
| `pipeline/tf2shlo_pipeline.cc` | legalize 编排 + Smuggle + 签名同步 |
| `wrap/pb2_unrank_wrap.cpp` | unrank 序列视图（一元 allowlist） |
| `wrap/pb2_strided_slice.cpp` | 动态切片直译 |
| `wrap/pb2_string_transport.cpp` | 编解码包裹正向 |
| `wrap/pb2_string_unwrap.cpp` | 编解码包裹回滚 |
| `pb2stablehlo.cpp` | CLI 入口 |
| `tests/unit/transform/` | 7 个单测套件 |
| `tests/integration/tf_transform/` | 集成 pytest（5 文件） |

## 附录 B 修订记录

| 批次 | 内容 | 状态 |
| --- | --- | --- |
| T1 | common 层 + 构建骨架 + 单测（~1070 行） | 已合入 |
| T2 | 三层推导（~840 行） | 已合入 |
| T3 | GraphDef 导入（~875 行；pb 域单测随 T8 补） | 已合入 |
| T4 | 12 个 MLIR pass（~1570 行） | 已合入 |
| T5 | restore + emit（~730 行；RunPipelines 拆 TU 随 T8 闭合） | 已合入 |
| T6 | wrap shape 组；决策：dyn-view 不迁、二元族移出（~890 行含测试） | 已合入 |
| T7 | wrap string 组；检视修复 axis 路径/DynamicStitch 布局/Squeeze 守卫（~1090 行含测试） | 已合入 |
| T8 | 管线编排 + 入口 + 链接闭合 + 集成 pytest（~700 行含测试） | 已合入 |
