# StableHLO sink 层（component LLD）

- 状态：组件层已实现（批 2）；pass 驱动与工具随后批（见「批次计划」）
- 源码基线：`annc/{include,lib}/Conversion/StablehloToAtir/`（根 = 框架无关层，`tf/` = TF 适配层）
- 触发条件：仅 `ANNC_ENABLE_TF2ATIR_V2=ON` 编译，默认构建零影响

## 链路位置

```
TF GraphDef / SavedModel
  └─ stablehlo 前端产物（stablehlo 主图 + TF 残差能力 op + shape/arith 脚手架）
       └─ [residue-sink]  ← 本层
            └─ ATIR 混合 IR（atir.*_v2 + 保留的 stablehlo/其他方言）
                 └─ V2 fusion / ATIR 下游 lowering / TF 运行时分区回退
```

## 归属：TF 前端适配层

目录上已物理分层：根目录 = 框架无关层（sink 驱动、`ResidueProvider` 接缝、
`V2OpMatcher`，换框架加子目录、根文件零改动），`tf/` = TF 适配层（kOpMap、
属性解码）。sink 驱动与接口不引用任何 TF 实现——换框架 = 新增 provider 实现。

载体已中性化：前端在 odml 转换完成后把残留载体 `tf.residue` 改为
`annc.residue`，sink 以 F2 形态（未注册 + `call_target_name`，与名字无关）
识别。两项保留的 TF 痕迹属契约而非泄漏：`tf.` 前缀在 wheel 阶段是生存前提
（其 unknown-op 路径对非 `tf.` 载体崩溃，改名必须后置）；`atir.unsupported`
的 `opType` 值为宿主回接所需（converter / 运行时按源 op 名回查）。

## 残差形态（TF，两形态）

| 形态 | 识别 | opType 来源 |
| --- | --- | --- |
| F1 | `stablehlo.custom_call`（callee 指向 TF op） | `call_target_name`；属性取解码后的 `backend_config` |
| F2 | 未注册、带 `call_target_name` 声明（`annc.residue` 载体即此形态） | 声明值 |

## 组件分工

- `ResidueProvider`：前端适配接缝。sink 驱动（批 3）负责转换、类型桥、回退与
  统计；provider 负责残差识别、op 表、属性解码与发射。
- `tf/TfResidueProvider`：TF 实现。`kOpMap` 是唯一映射契约（complement-only：
  sparse / resource / string / top_k / constant / view 域）；`Concat` 因
  `concat_dim` 与 `axis` 的位置差异被布局门排除；`emitConstant` 下沉
  DT_STRING 常量到 `constant_v2`（`values` 载荷）。
- `V2OpMatcher`：纯查表解析目标（mnemonic 须已注册）；属性按 ODS 声明名组装
  （别名→精确名→camel/snake，IntegerAttr 统一 i64）；本地 verify，失败回退
  `atir.unsupported`（哨兵填充仅限 `sparse_segment_*` 的 `num_segments`）。

## 失败矩阵

| 情形（批 3 驱动判定） | 结果 |
| --- | --- |
| 残差不在 `kOpMap` / 目标未注册 | `atir.unsupported`，`reason=unknown_op` |
| 发射后本地 verify 仍失败（哨兵修复后） | `atir.unsupported`，`reason=spec_mismatch` |

`detail` 字段现状约定：恒为空串。provider 接口（match / emitMapped /
hasMapping）不回传发射拒绝原因，driver 无从填写具体证据；下游路由只依
赖 `reason`。待 provider 扩展拒绝原因查询后再回填，届时同步本表。
| 已注册非残差 op | 原位保留，计入 census（`keep`） |

`atir.unsupported` 字段：`opType` + 原始属性 `metadata` + `reason` + `detail`，
保真承接，不丢信息。

## 值域

V2 op 一律使用 builtin 张量（`AnyTensor`）；`!atir.tensor` 属 V1 值域。
string 张量 = builtin 张量 + `!atir.string` 元素类型（无 encoding）。

## 兼容方言 `tf_type`

仅让携带 TF 类型的 artifact 无损解析：任意 `!tf_type.<tag>`（含
`resource<tensor<...>>` 等参数）可 parse，功能上只识别 `string`
（`isTfStringType`）。不参与语义，类型归一化由 sink 类型转换器完成（批 3）。
`#tf_type.shape` 属性解析随后续批次按需补（当前 artifact 不携带）。

## 测试策略

| 层 | 方式 | 状态 |
| --- | --- | --- |
| 组件契约 | GTest（`stablehlo_sink_component_test.cpp`，6 用例） | 已实现（批 2） |
| pass 端到端 | FileCheck（`tf-residue-v2-sink.mlir` 经 `annc-tf2atir-v2`） | 随工具批 |
| 数值等价 | `tests/e2e/tf/e2e_kp_embedding_patterns.py` | 随工具批 |

## 批次计划

| 批次 | 内容 | 状态 |
| --- | --- | --- |
| 批 3 | `residue-sink` 驱动（ConversionTarget / TypeConverter / pattern 优先级 / census）、`Passes.td`、pass 构造与统计输出 | 已合入 |
| 批 4 | `annc-tf2atir-v2` 工具、端到端 FileCheck（`tf-residue-v2-sink.mlir`）、CLI 文档 | 已合入（`--op-fusion-v2` 选项随 V2 融合线另行批次） |
