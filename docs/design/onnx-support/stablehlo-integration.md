# ONNX 经 StableHLO 接入 ANNC

## 1. 目标与设计边界

让 ANNC 从 StableHLO 中找到 BMMAdd、TensordotMatMul、MaskedMul 和 LayerNorm，把每组计算编译成一个 kernel，由 ONNX Runtime（ORT）在当前会话内执行。沿用已有 AArch64 后端、kernel 和 EP。原 ONNX 文件与外部权重保持原样。

## 2. 总体流程

`annc-onnx-pipeline --model model.onnx` 是两条路径的入口。默认经 ATIR 导入和融合。设置 `ANNC_ONNX_USE_STABLEHLO=1` 后，编译按下表执行。四类算子的 `ANNC_EP_FUSION_*` 开关独立生效。

| 阶段 | 工具 | 结果 |
| --- | --- | --- |
| 转换并记录来源 | `annc-onnx2stablehlo` | `model.stablehlo.mlir`，保存主图、原图索引和来源属性 |
| 在 StableHLO 上匹配融合 | `annc-stablehlo-opt --annc-stablehlo-partition` | `model.partitioned.mlir`，保留主图并附上候选函数 |
| 绑定原图区域 | `annc-stablehlo-opt --annc-stablehlo-bind-onnx` | `model.bound.mlir`，记录绑定结果及候选拒绝原因 |
| 提取 kernel | `annc-stablehlo-opt --annc-stablehlo-kernels` | `model.fused.mlirbc`，只含通过绑定的 kernel 及公共元数据 |
| 编译和发布 | `annc-asm`、`annc --shared` | `model.fused.mlir`，有站点时同时生成 `libannc_kernels.so` |
| 会话内接管 | `libannc_ep.so` | ORT 按成员节点替换内存图，EP 调用 kernel |

两条路径复用 `annc-asm --annc-fast-codegen --annc-aarch64-gemm-pipeline`。ATIR 发布 `sites.json`，StableHLO 发布含 `fusion.metadata` 的标准 kernel IR。配套转换和匹配工具使用同一套 MLIR，公共后端通过 IR 文件接收 kernel。[流水线实现](../../../frontends/onnx/onnx-pipeline/annc-onnx-pipeline.cpp) · [构建说明](../../guides/build-from-source.md#onnx-的-stablehlo-前端)

StableHLO 路径发布 kernel IR 前，使用与 EP 读取端相同的 `func`、`arith` 和 `LLVM` 方言解析。IR 含有未注册方言的操作或类型时，流水线在发布阶段报错。

## 3. ONNX 转换与来源记录

转换直接使用 EP 导出的 `model.onnx`，手工编译时使用 `--model` 指定的文件。首次导入读取一次 ONNX，使用同一个 `ModelProto` 生成 ONNX IR 和原图索引。节点名必须非空且在图内唯一。外部权重由 ONNX-MLIR 按模型目录加载，文件缺失时沿用导入报错。[导入与索引实现](../../../addons/onnxruntime/stablehlo/pipeline/annc-onnx-framework-opt.cpp)

转换驱动按顺序运行 ONNX 分解、形状推导、规范化和常量传播，再执行前处理、ONNX-MLIR 转换、后处理和来源收尾。ONNX 优化保留 `MatMul → Reshape → Add` 的输出边界和连续 Mul 的结构，供后续匹配识别。[转换驱动](../../../addons/onnxruntime/stablehlo/pipeline/annc-onnx2stablehlo.cpp)

| 转换场景 | 处理方式 |
| --- | --- |
| 字符串 Constant / Expand | 前处理用 i64 载体保留已知 rank 的形状和连线，字符串内容按字节存入配置。后处理生成带原来源的 `stablehlo.custom_call`，留在主图由 ORT 执行。 |
| Slice 与动态 Tile | 前处理补齐 Slice 的默认参数并统一索引，对动态 Tile 写入临时标记。后处理用 StableHLO 广播和 reshape 表达动态 Tile，并将支持的形状计算转成 StableHLO。 |
| 残留算子 | 支持的普通残留算子继续降为 StableHLO。ORT 私有 `com.microsoft.FusedTensordotMatMul` 保留为 `atir.opaque`，携带原类型、属性和来源。它的输出可以作为邻近融合的输入，私有节点本身留给 ORT 执行。未知残留或转换后仍无 rank 的类型报错。 |

含 `atir.opaque` 的文件由配套 ANNC 工具读取。纯 StableHLO 的 portable 序列化单独验证。[转换与校验代码](../../../addons/onnxruntime/stablehlo/passes)

### 3.1 节点和张量的对应关系

| IR 属性 | 保存内容 |
| --- | --- |
| 模块的 `annc.onnx_model` | 导入时的原图索引（`annc-model-index/2`），包含节点、连线、张量角色和类型 |
| 操作的 `source_ids` | 从 MLIR location 得到的原 ONNX 节点名，可以是一组节点 |
| 参数的 `annc.tensor_names` | 函数参数对应的原张量名 |
| 操作的 `annc.result_tensors` | 每个结果对应的原张量名及已确认的别名 |

来源收尾确认公共输入输出的对应关系，并为可证明的中间值补充别名，包括被消除的 Identity。辅助操作的来源可以为空，匹配候选真正用到的原图边界由绑定确认。对应关系随 IR 传递，提取 kernel 时移除主图、索引和临时属性。[来源收尾](../../../addons/onnxruntime/stablehlo/passes/postprocessing/onnx_source_finalize.cpp)

### 3.2 转换后权重名称丢失

权重常量可能仍在 IR 中，但其 `Initializer_<tensor>` location 被折叠或合并。来源收尾先用已有 location。缺少 location 时，从已确认的计算出口检查常量操作数及其位置，要求对应原参数为不可变 initializer，类型、已知形状相容，且关系可唯一确认。多个相同权重可以映射到同一个常量，绑定时依据原图实际入边选择名称。可覆盖的 initializer、缺少证据的固定参数和已消失的常量值均无法通过这项绑定。[权重来源恢复](../../../addons/onnxruntime/stablehlo/passes/postprocessing/onnx_source_finalize.cpp) · [原图绑定](../../../frontends/onnx/converter/BindStablehlo.cpp)

## 4. 匹配、绑定与编译

### 4.1 在 StableHLO 上匹配

各模式沿值的生产者与消费者关系识别实际计算，结合类型、归约轴、广播维度和动态形状表达式确认边界。匹配成功后，在模块中附加含 `stablehlo.custom_call` 的融合函数，并保留主图供绑定。模式匹配只使用当前 IR，原图索引在下一阶段使用。[模式实现](../../../frontends/stablehlo/Patterns) · [区域发布](../../../frontends/stablehlo/HloGraph.cpp)

### 4.2 ONNX 绑定的可切判定

绑定读取 IR 中的索引和来源属性，确认用 kernel 替换原 ONNX 区域后计算和连线仍完整。

1. 从 HLO 输出走到输入边界，核对实际区域、类型及融合函数记录。
2. 确认唯一的原图输出，再沿原图连线回溯到输入边界和不可变 initializer，得到实际要替换的成员。
3. 确认成员非空、与其他已绑定区域无重叠，固定参数来自不可变 initializer，每条外部数据入边都有对应槽位。
4. 确认区域外还会使用的结果均由 kernel 输出。原节点有多段 HLO 数值计算时，检查仍活跃的部分是否已完整覆盖。
5. 确认部署条件。BMMAdd 的数据输入须有已知的 ONNX rank，LayerNorm 的数据输入须有原图生产节点，ORT 私有算子继续由框架执行。

候选通过后，原成员名写到 FusionInfo 的 `member_node_names`，原张量名写到 `args/outputs` 的 `tf_name`，并保留类型和参数角色。不满足条件的候选记录拒绝原因，原节点继续由 ORT 调度。[绑定实现](../../../frontends/onnx/converter/BindStablehlo.cpp)

### 4.3 复用公共后端

提取只保留通过绑定的函数，将 custom-call 转成公共后端可读的标准 tensor 调用，再由现有后端完成入口转换、bufferization 和适用的 GEMM lowering。kernel 入口沿用 `annc_context_v3`，通过 `AnncKernelContext` 传入参数和输出。[kernel 提取](../../../frontends/stablehlo/Partition.cpp) · [入口转换](../../../annc/lib/Conversion/Context/LowerContextKernelEntries.cpp)

BMMAdd 可在编译期把符合广播和常量条件的前置 Mul、可选 Add 折入 kernel 私有的权重和偏置。原图仍绑定原有三个参数槽位，公共 lowering 只覆盖该 kernel 使用的权重和偏置视图，packed-B 读取同一份准备后的权重。计算沿用原 kernel，以浮点容差验收。[BMMAdd 参数准备](../../../frontends/stablehlo/Patterns/BMMAdd.cpp)

## 5. ORT 会话内替换和执行

EP 对 ORT 当前图导出 ONNX 后，按既有缓存流程调用 `annc-onnx-pipeline`。前端和工具信息进入缓存键，默认 ATIR 和 StableHLO 可共用缓存根目录。

EP 从 `model.fused.mlir` 的 `fusion.metadata` 读取原成员、边界张量和 ABI，通过 `AddNodesToFuse` 申报。ORT 在会话内创建融合节点，EP 绑定参数、输出和共享库符号，并通过现有回调运行 kernel。kernel IR 缺失时使用原有 `sites.json` 接口。已有 kernel IR 损坏时，本次编译结果被拒绝。零站点保留有效的元数据产物，省略共享库。[缓存和元数据读取](../../../addons/onnxruntime/annc_ep_compile.cc) · [EP 接管与执行](../../../addons/onnxruntime/annc_ep.cpp)

成功后默认清理转换和 lowering 的临时文件。`--keep-temps` 可保留中间 IR 和转换诊断 `model.stablehlo.json`。正式的 `model.fused.mlir` 保留给 EP 使用。[CLI 工作流](../../guides/cli-workflows.md#onnx-的-stablehlo-前端)

## 6. 当前支持范围

| 模式 | 当前范围 |
| --- | --- |
| BMMAdd | FP32 rank-3/rank-4，矩阵行维 M 可动态，batch/K/N 静态。weight 和 bias 为不可变参数。 |
| TensordotMatMul | FP32 rank-3 data 与不可变 rank-2 weight，前两维可动态，K/N 静态。动态形状沿用原 kernel 的非溢出计算约定。 |
| LayerNorm | FP32 rank-2，按静态末维归一化，首维可动态，gamma/beta/epsilon 为不可变参数。 |
| MaskedMul | FP32 rank-3 data 与 i64 rank-2 ids，前两维对齐。支持现有注册的归约和多输出组合，沿用 kernel 的固定累加顺序。 |
