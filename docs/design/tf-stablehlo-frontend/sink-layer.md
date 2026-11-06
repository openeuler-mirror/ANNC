# StableHLO sink 层（component LLD）

- 状态：组件层已实现（批 2）；pass 驱动与工具随后批落地（见「批次计划」）
- 源码基线：`annc/{include,lib}/Conversion/StablehloToAtir/`
- 触发条件：仅 `ANNC_ENABLE_STABLEHLO=ON` 编译，默认构建零影响

## 链路位置

```
TF GraphDef / SavedModel
  └─ stablehlo 前端产物（stablehlo 主图 + TF 残差能力 op + shape/arith 脚手架）
       └─ [residue-sink]  ← 本层
            └─ ATIR 混合 IR（atir.*_v2 + 保留的 stablehlo/其他方言）
                 └─ V2 fusion / ATIR 下游 lowering / TF 运行时分区回退
```

## 残差三形态（TF）

| 形态 | 识别 | opType 来源 |
| --- | --- | --- |
| F1 | 未注册 `tf.*` generic op | op 名去前缀 |
| F2 | `stablehlo.custom_call`（callee 指向 TF op） | `call_target_name`；属性取解码后的 `backend_config` |
| F3 | 未注册、带 `call_target_name` 声明的 generic op | 声明值 |

## 组件分工

- `ResidueProvider`：前端适配接缝。sink 驱动（批 3）负责转换、类型桥、回退与
  统计；provider 负责识别、op 表、属性解码与发射。
- `TfResidueProvider`：TF 实现。`kOpMap` 是唯一映射契约（complement-only：
  sparse / resource / string / top_k / constant / view 域）；未在表内的 op 回退
  `atir.unsupported`。
- `V2OpMatcher`：纯查表解析目标（mnemonic 须已注册）；属性按 ODS 声明名组装
  （别名→精确名→camel/snake 转换，IntegerAttr 统一 i64）；发射后本地 verify，
  失败回退 `atir.unsupported`（含哨兵填充的一次修复）。

## 失败矩阵

| 情形（批 3 驱动判定） | 结果 |
| --- | --- |
| 残差不在 `kOpMap` / 目标未注册 | `atir.unsupported`，`reason=unknown_op` |
| 残差 rank 未定（rank gate） | `atir.unsupported`，`reason=unrank_blocked` |
| 发射后本地 verify 仍失败（哨兵修复后） | `atir.unsupported`，`reason=spec_mismatch` |
| 已注册非残差 op | 原位保留，计入 census（`keep`） |

`atir.unsupported` 字段：`opType` + 原始属性 `metadata` + `reason` + `detail`，
下游按 TF 运行时分区语义处理（保真承接，不丢信息）。

## 值域

V2 op 一律使用 builtin 张量（`AnyTensor`）；`!atir.tensor` 属 V1 值域。string
张量 = builtin 张量 + `!atir.string` 元素类型（无 encoding）。

## 兼容方言 `tf_type`

仅让携带 TF 类型的 artifact 无损解析：任意 `!tf_type.<tag>` 可 parse，功能上
只识别 `string`（`isTfStringType`）。不参与语义，类型归一化由 sink 类型转换器
完成（批 3）。`#tf_type.shape` 属性解析随后续批次按需补（当前 artifact 不携带）。

## 测试策略

| 层 | 方式 | 状态 |
| --- | --- | --- |
| 组件契约 | GTest（`tests/unit/annc/stablehlo_sink_component_test.cpp`，5 用例） | 已实现（批 2） |
| pass 端到端 | FileCheck（`tf-residue-v2-sink.mlir` 经 `annc-tf2atir-v2`） | 随工具批（批 4） |
| 数值等价 | `tests/e2e/tf/e2e_kp_embedding_patterns.py` | 随工具批（批 4） |

## 批次计划与已知差距

| 批次 | 内容 |
| --- | --- |
| 批 3 | `residue-sink` pass 驱动（ConversionTarget / TypeConverter / pattern 优先级 / rank gate / census）、`Passes.td`、`createStablehloSink`/`printSinkStat` |
| 批 4 | `annc-tf2atir-v2` 工具、端到端 FileCheck、CLI 文档 |

已知差距（恢复条件见迁移记录）：`#tf_type.shape` 解析、`tf.Const` string 下沉
与 `constant_v2.values` 契约、V2 融合路径的 `CustomizeOp` 跨域处理（方案 B
`atir.customize_v2` 优先，兜底放宽 V1 契约 + 值域互斥校验）。rank refinement
（rankinfer）不在本层范围：未定 rank 的残差由 rank gate 归类 `unrank_blocked`。
