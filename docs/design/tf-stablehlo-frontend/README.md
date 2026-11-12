# ANNC TF StableHLO 前端与 sink 层设计（模块文档集）

| 文档控制项 | 内容 |
| --- | --- |
| 文档编号 | ANNC-TFSTABLEHLO-LLD-000 |
| 版本与日期 | 1.0，2026-11-12；V2 stablehlo 路线生产端（pb2stablehlo）与消费端（residue-sink）交付后的设计组索引 |
| 文档状态 | 索引。各模块设计见同目录子文档 |
| 源码基线 | `addons/tensorflow/transform/`、`annc/{include,lib}/Conversion/StablehloToAtir/`、`annc/tools/annc-tf2atir-v2/` |
| 上级文档 | [ANNC 总体架构](../../architecture/overview.md)（V2 stablehlo 路线一节） |
| 关联文档 | [架构决策索引](../../decisions/README.md)（ADR-023：wheel 隔离构建域）、[CLI 工作流](../../guides/cli-workflows.md)、[环境变量参考](../../guides/environment-variables.md) |

## 1 本文档集的定位

V2 stablehlo 路线的两个核心子系统：前端把 TF GraphDef 转成 stablehlo 主图
加残差载体的产物，sink 层把残差下沉为 ATIR 供 V2 融合与下游 lowering 消费。
两份文档以产物契约为交接点（残差 F1/F2 两形态、wrap 边界属性通道），
跨模块协议由定义方说明语义，由消费方说明识别与映射规则。

```text
TF GraphDef / SavedModel
  └─ [pb2stablehlo 前端]（frontend-transform.md）
       └─ stablehlo 主图 + 残差载体 + shape/arith 脚手架（.mlir）
            └─ [residue-sink]（sink-layer.md）
                 └─ ATIR 混合 IR → V2 fusion / ATIR 下游 lowering
```

## 2 文档清单

| 文档 | 覆盖范围 | 主要源码落点 |
| --- | --- | --- |
| [frontend-transform.md](frontend-transform.md) | pb2stablehlo 前端：pb 域改写、三层推导、MLIR pass 集、残差封装与恢复、wrap 层（shape/string）、管线编排、入口与失败模型 | `addons/tensorflow/transform/` |
| [sink-layer.md](sink-layer.md) | residue-sink：sink 驱动、残差形态识别（F1/F2）、`kOpMap` 映射契约、`V2OpMatcher`、`tf_type` 兼容方言、census 统计 | `annc/{include,lib}/Conversion/StablehloToAtir/` |

## 3 状态与阅读入口

- 两份 LLD 状态头均声明源码基线与差距落点；当前行为以源码与测试为准。
- 端到端入口：`pb2stablehlo`（前端产物）→ `annc-tf2atir-v2`（残差下沉），
  CLI 契约见 [CLI 工作流](../../guides/cli-workflows.md)。
- 构建前提：`ANNC_ENABLE_TF2ATIR_V2=ON`（默认 OFF，默认构建零影响）；
  前端另需 TF 2.20 pip wheel 与 llvm 头来源，见
  [从源码构建](../../guides/build-from-source.md)。
- A/B 量测开关（`ANNC_NO_*`）见
  [环境变量参考](../../guides/environment-variables.md)。

## 4 规划中的扩展

| 项 | 说明 | 载体 |
| --- | --- | --- |
| V2 融合线（op-fusion-v2） | sink 之后按 `--op-fusion-v2` 开启的站点融合（outline 为 `annc_context_v3` kernel func）；垂直切片迁移中，首切 pattern 为 kp_gather | `annc/lib/Dialect/Atir/Passes/{OpFusionV2.cpp,FusionV2/}` |
