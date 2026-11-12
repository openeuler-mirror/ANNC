# StableHLO sink 层设计（residue-sink）

| 文档控制项 | 内容 |
| --- | --- |
| 文档编号 | ANNC-TFSINK-LLD-001 |
| 版本与日期 | 1.0，2026-11-12 |
| 文档状态 | 模块设计。描述当前设计意图与实现遵循度，差距见第 8 节 |
| 评审对象 | ANNC 架构师、TF 前端与 sink 层开发、V2 融合与后端开发、测试负责人 |
| 源码基线 | `annc/{include,lib}/Conversion/StablehloToAtir/`（10 个文件，基线 `810dd12`，分支 `feature/transform-final`） |
| 上级文档 | [总体架构](../../architecture/overview.md)（V2 stablehlo 路线一节）、[tf-stablehlo-frontend 设计组](README.md) |
| 关联文档 | [前端设计（pb2stablehlo）](frontend-transform.md)（产物生产方）、[ADR-023](../../decisions/decisions.md#adr-023)、[CLI 工作流](../../guides/cli-workflows.md) |

## 1 概述

### 1.1 定位与作用

本组件是 V2 stablehlo 路线的**消费端**：把前端产物（stablehlo 主图 + TF 残差能力 op + shape/arith 脚手架）中的残差下沉为 ATIR（`atir.*_v2`），其余方言原样保留，产出 ATIR 混合 IR，供 V2 融合与 ATIR 下游 lowering、TF 运行时分区回退使用。

```text
TF GraphDef / SavedModel
  └─ stablehlo 前端产物
     └─ 生产端：[pb2stablehlo 前端](frontend-transform.md)
       └─ [residue-sink]  ← 本层
            └─ ATIR 混合 IR（atir.*_v2 + 保留的 stablehlo/其他方言）
                 └─ V2 fusion / ATIR 下游 lowering / TF 运行时分区回退
```

### 1.2 设计目标与优先级

| 编号 | 目标 | 优先级 | 判据 |
| --- | --- | --- | --- |
| G1 | 保真承接：残差语义完整映射或显式声明不支持，不丢信息 | 最高 | `atir.unsupported` 携带 `opType` + 原始属性 + `reason`；组件测试锁定契约 |
| G2 | 框架无关：sink 驱动与接口不引用任何 TF 实现 | 最高 | 换框架 = 新增 provider 子目录实现，根文件零改动 |
| G3 | 单一映射契约：`kOpMap` 是 TF op → ATIR 目标的唯一事实来源 | 最高 | complement-only 域外 op 一律 `atir.unsupported`（fail-closed） |
| G4 | 与前端契约解耦：识别与载体名拼写无关 | 高 | F1/F2 两形态按属性识别（§3.1），兼容改名（S3） |
| G5 | 默认构建零影响 | 高 | 仅 `ANNC_ENABLE_TF2ATIR_V2=ON` 编译 |

### 1.3 范围

在范围内：sink 驱动（`StablehloSink`，ConversionTarget/TypeConverter/ pattern 优先级/census 统计）、`ResidueProvider` 接缝、`V2OpMatcher` 查表解析、TF 适配层（`kOpMap`、属性解码与发射）、`tf_type` 兼容方言、
`annc-tf2atir-v2` 工具。

范围之外：残差的产生（见[前端设计](frontend-transform.md)）、ATIR 下游 lowering 与 V2 融合、TF 运行时分区回退。

### 1.4 术语

| 术语 | 含义 |
| --- | --- |
| 残差 | 前端无法转成 stablehlo 的 TF 能力 op，经载体形态传递到本层 |
| F1 / F2 形态 | 残差的两种 IR 形态（§3.1），sink 均支持 |
| `kOpMap` | TF op 名 → ATIR 目标的唯一映射契约，complement-only 域 |
| census | sink 结束后的去向统计：sunk（已下沉）/ unsupported（回退哨兵）/ kept（原位保留）三类计数 |
| `tf_type` 兼容方言 | 仅使携带 TF 类型的产物无损解析的兼容层，不参与语义 |

## 2 设计约束、假设与原则

| 约束 | 内容 | 违反后果 |
| --- | --- | --- |
| C1 框架无关 | 根目录（sink 驱动、`ResidueProvider` 接缝、`V2OpMatcher`）不引用 TF 实现 | 分层失效，换框架需改根文件 |
| C2 映射契约唯一 | 新增映射只改 `kOpMap`；驱动与 matcher 不持有个案分支 | 映射事实分散，契约不可审计 |
| C3 值域边界 | V2 op 一律 builtin 张量（`AnyTensor`）；`!atir.tensor` 属 V1 值域，不混用 | 类型转换器两值域混装 |
| C4 识别与名字解耦 | F2 识别依据是未注册 + `call_target_name`，与载体名拼写无关 | 前端改名（S3）破坏识别 |
| C5 不动 stablehlo 主体 | sink 只下沉残差，stablehlo 主图与脚手架方言原样保留 | 重复 lowering 或语义漂移 |

设计原则：映射失败是预期路径而非异常——每个残差要么映射、要么以带原因的 `atir.unsupported` 显式落地；驱动不因个别残差失败而中断整图转换。

## 3 总体设计

### 3.1 残差形态（TF，两形态）

| 形态 | 识别 | opType 来源 |
| --- | --- | --- |
| F1 | `stablehlo.custom_call`（callee 指向 TF op） | `call_target_name`；属性取解码后的 `backend_config` |
| F2 | 未注册、带 `call_target_name` 声明（`annc.residue` 载体即此形态） | 声明值 |

前端在 odml 转换完成后将 `tf.residue` 统一改名为 `annc.residue`（[前端设计](frontend-transform.md) §4.2，S3 时序约束），sink 以 F2 形态识别、与拼写无关。两项保留的 TF 痕迹属契约而非泄漏：`tf.` 前缀在 wheel 阶段是生存前提（其 unknown-op 路径对非 `tf.` 载体崩溃，改名必须后置）；`atir.unsupported` 的 `opType` 值为宿主回接所需（converter / 运行时按源 op 名查询）。

### 3.2 组件分工

| 组件 | 职责 |
| --- | --- |
| `StablehloSink`（驱动） | 转换、类型桥、回退与统计； ConversionTarget / TypeConverter / pattern 优先级 / census |
| `ResidueProvider`（接缝） | 残差识别、op 表、属性解码与发射（match / emitMapped / hasMapping） |
| `tf/TfResidueProvider`（TF 实现） | `kOpMap` 契约持有方；`Concat` 因 `concat_dim` 与 `axis` 的位置差异被布局门排除；`emitConstant` 下沉 DT_STRING 常量到 `constant_v2`（`values` 载荷） |
| `V2OpMatcher`（查表解析） | mnemonic 须已注册；属性按 ODS 声明名组装（别名→精确名→camel/snake，IntegerAttr 统一 i64）；本地 verify，失败回退 `atir.unsupported`（哨兵填充仅限 `sparse_segment_*` 的 `num_segments`） |
| `TfTypeCompat` | `tf_type` 兼容方言（§3.4） |

### 3.3 工具与产物

`annc-tf2atir-v2 input.mlir -o output.mlir`（省略 `-o` 时输出到 stdout），行为契约见 [CLI 工作流](../../guides/cli-workflows.md)。`--op-fusion-v2` 选项随 V2 融合线另行批次。

### 3.4 兼容方言 `tf_type`

仅使携带 TF 类型的产物无损解析：任意 `!tf_type.<tag>`（含 `resource<tensor<...>>` 等参数化形态）可 parse，功能上只识别 `string`（`isTfStringType`）。不参与语义，类型归一化由 sink 类型转换器完成。`#tf_type.shape` 属性解析按需补（当前产物不携带）。

## 4 核心机制

### 4.1 值域

V2 op 一律使用 builtin 张量（`AnyTensor`）；`!atir.tensor` 属 V1 值域。 string 张量 = builtin 张量 + `!atir.string` 元素类型（无 encoding）。

### 4.2 census 统计

sink 结束后遍历统计三类去向：sunk（已映射下沉）、unsupported（回退哨兵）、kept（原位保留）。kept 范围仅限脚手架方言（stablehlo/shape/tensor 等）；被 provider 认领的残差不计入 kept。

## 5 关键设计决策

| 决策 | 理由 | 被否备选 | 影响 |
| --- | --- | --- | --- |
| rank gate 移除 | 前端已产 rank 契约（残差载体必带 rank），sink 侧二次 gate 属重复防线 | 保留双保险 | unranked 残差直接映射（`MapsUnrankedResidueThrough` 锁定） |
| dual-presence 审计移除 | 已禁止恢复；残差单次决定去向，无双写形态 | 保留审计 | 简化 census 语义 |
| `detail` 字段恒空 | provider 接口（match / emitMapped / hasMapping）不回传发射拒绝原因，驱动无从填写证据；下游路由只依赖 `reason` | 扩展接口回传原因 | 待 provider 扩展后回填，届时同步 §6 |
| `Concat` 布局门排除 | `concat_dim`（v1 首位）与 `axis`（v2 末位）位置差异，静态不可证等价 | 按版本分别映射 | Concat 保持残差，由宿主回接 |

## 6 失败模型

| 情形（驱动判定） | 结果 |
| --- | --- |
| 残差不在 `kOpMap` / 目标未注册 | `atir.unsupported`，`reason=unknown_op` |
| 发射后本地 verify 仍失败（哨兵修复后） | `atir.unsupported`，`reason=spec_mismatch` |
| 已注册非残差 op | 原位保留，计入 census（`keep`） |

`atir.unsupported` 字段：`opType` + 原始属性 `metadata` + `reason` + `detail`，保真承接，不丢信息。`detail` 现状恒为空串（§5）。

## 7 质量属性与验证

| 层 | 资产 | 覆盖 | 命令 |
| --- | --- | --- | --- |
| 组件契约 | `tests/unit/annc/stablehlo_sink_component_test.cpp`（7 用例） | F1/F2 契约、kOpMap 全目标可解析、V2 op / 常量发射契约、tf_type 解析 | `ctest --test-dir build -R StablehloSink` |
| pass 端到端 | `stablehlo_sink_pass_test.cpp`（7 用例）+ FileCheck（`tf-residue-v2-sink.mlir`，17 断言，经 `annc-tf2atir-v2`） | F1/F2 下沉、unknown 回退、脚手架保留、string 类型桥、unranked 映射、spec_mismatch 回退 | 同上 / `bash tests/FileCheck/run.sh Conversion` |
| 数值等价 | `tests/e2e/tf/e2e_kp_embedding_patterns.py` | 原图与重写图输出对照 | 见文档头说明 |

改动纪律：`kOpMap` 增删必须同步组件测试的"全目标可解析"用例；新增哨兵填充域须补 spec_mismatch 反例。

## 8 现状与目标差距

| 编号 | 差距 | 影响 | 计划 |
| --- | --- | --- | --- |
| GAP-1 | `detail` 恒空（provider 接口不回传拒绝原因） | 排障只到 `reason` 粒度 | provider 扩展拒绝原因查询后回填 |
| GAP-2 | `#tf_type.shape` 属性解析未实现 | 携带该属性的产物无法解析（当前产物不携带） | 按需补 |
| GAP-3 | `--op-fusion-v2` 未接（V2 融合线范围） | 工具不能驱动 V2 融合 | 随 V2 融合批次 |

## 附录 A 源码位置索引（文件级）

| 文件 | 职责 |
| --- | --- |
| `include/.../StablehloSink.h`、`lib/.../StablehloSink.cpp` | sink 驱动（转换/类型桥/census） |
| `include/.../ResidueProvider.h` | 前端适配接缝接口 |
| `include/.../V2OpMatcher.h`、`lib/.../V2OpMatcher.cpp` | 目标解析与属性组装 |
| `lib/.../TfTypeCompat.cpp` | `tf_type` 兼容方言 |
| `lib/.../tf/TfResidueProvider.cpp` | TF 实现（`kOpMap` / 解码 / 发射） |
| `include/.../Passes.td` | pass 声明 |
| `tests/unit/annc/stablehlo_sink_{component,pass}_test.cpp` | 组件 / pass 测试 |
| `tests/FileCheck/Conversion/tf-residue-v2-sink.mlir` | 端到端 FileCheck |

## 附录 B 修订记录

| 批次 | 内容 | 状态 |
| --- | --- | --- |
| 批 1 | stablehlo 方言集成与值域（EXCLUDE_FROM_ALL、属性快照/还原） | 已合入 |
| 批 2 | `ResidueProvider` 接缝、TF 适配层（`kOpMap`）、`V2OpMatcher`、`tf_type` 兼容方言 + 组件 GTest | 已合入 |
| 批 3 | `residue-sink` 驱动（ConversionTarget / TypeConverter / pattern 优先级 / census）、`Passes.td`、pass 构造与统计输出 | 已合入 |
| 批 4 | `annc-tf2atir-v2` 工具、端到端 FileCheck、CLI 文档 | 已合入 |
