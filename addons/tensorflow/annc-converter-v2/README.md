# annc-converter-v2

`annc-converter` 的并行重实现：把 Fusion Metadata 应用到原始 TensorFlow
GraphDef 上，将被融合子图替换为 `ANNCFused` 自定义 Op 节点。与 v1 的
有意差异：

- **元数据来源仅支持 `fusion_metadata.json`**（`annc-fusion-metadata`
  先从融合 ATIR 导出）；v1 的 ATIR 文本直读路径不进入 v2；
- **不可落地的站点整条丢弃**（宁缺不错，stderr 告警）：输出端点不在
  原图、输入无法解析、常量型输入接到非常量节点、控制边指向被替换
  节点、站点间循环接线；全部丢弃时保留原图并返回成功；
- **仅支持 TensorFlow 2.x**；SDK 的 protobuf 头需 ≥ v3.22 的
  `ErrorCollector` 签名；
- 被替换的输出节点**始终**生成 Identity 别名（v1 在仍有消费者时跳过）。

## 部署形态：PRELOAD 插件

产物为 `libannc_graph_rewriter.so`：TF/protobuf 符号保持未定义，由
predictor 进程加载插件时解析——构建与部署**不依赖 pip TensorFlow**。`ANNC_V2_REWRITE_INPROC=1` 时
`ANNCOptimizer` 在 Grappler 时机通过 `dlopen` 进程内调用改写入口，失败
自动回退 v1 pipeline 路径（env 见 [部署文档](../../../tests/tf-build-and-deploy.md)）。

## 与 v1 的差异：TF/LLVM 进程隔离

v1 同时链接 TF protobuf 与 annc 的 LLVM/MLIR。本工具的形态：

- **本工具进程只含 TensorFlow**：TF 头与 `libtensorflow_framework`
  来自宿主 TF 依赖，不链接任何 annc LLVM/MLIR 目标；
- **annc 工具链进程只含 LLVM/MLIR**：跨进程只交换文件
  （`fusion_metadata.json` + GraphDef pb）；metadata 读取为纯 std::
  复刻（`fusion_meta.h`），只消费 `annc-fusion-metadata` 的显式 JSON
  契约，不解析 MLIR。

V2 路线的离线编排由 `annc-tf-pipeline-v2`
（`frontends/tensorflow/tf-pipeline-v2/`）承担：它按
pb2stablehlo → annc-stablehlo-sink → 融合 → fast-codegen →
`annc-fusion-metadata` 的顺序产出本组件所需的
`fusion_metadata.json` 与 fusion-only ATIR。

## 离线形态：annc-converter-v2 可执行

与插件同一份 core objects（`rewriteGraphDefInPlace`），以 CLI 形态离线
改写 GraphDef：

```shell
annc-converter-v2 --metadata_json fusion_metadata.json \
  --input_graphdef model.pb --output_graphdef rewritten.pb \
  [--shared_lib_path kernel.so] [--atir_module_path fused.mlir]
```

`--shared_lib_path` 与 `--atir_module_path` 可同时给出：AOT 站点读共享
库、JIT 站点读 ATIR 模板，各取所需。**依赖**：链接构建时解析出的 TF
framework（默认 pip wheel，或 `ANNC_TENSORFLOW_LIBRARIES` 显式指定）；
**SDK headers-only 模式下该可执行被跳过**（无可链库，只构建 PRELOAD
插件）——此时改写交给 predictor 进程内插件，编排工具用
`annc-tf-pipeline-v2 --skip-converter` 跳过本步骤。产物 `rewritten.pb`
仅供人工检查，不参与 predictor 运行时。

## 构建与运行

构建命令见
[从源码构建 · annc-converter-v2](../../../docs/guides/build-from-source.md)。
部署机（standalone 插件构建）只依赖显式宿主 SDK 头，不探测任何 TF 库；
`ANNC_DEBUG=ON` 与本组件不兼容（其全局 `-UNDEBUG` 会破坏 TF 头的链接，
配置期报错）。

## 进程内改写的配置

`ANNCOptimizer`（env 灰度）需要：

- `ANNC_V2_FUSION_METADATA`：`fusion_metadata.json` 路径（由
  `annc-fusion-metadata` 产出）；
- `ANNC_V2_SHARED_LIB` / `ANNC_V2_ATIR_MODULE` / `ANNC_V2_KERNEL_NAME`：
  kernel 产物路径（对应 metadata 中的 AOT/JIT fusion）。

改写后图保持原图输出名（被替换节点处生成 Identity 别名）；站点丢弃
路径保留原图计算。

## 验证与契约

单元测试在 `tests/unit/converter_v2/`（GTest/CTest；standalone 注册在
本目录 CMakeLists）：端点解析、metadata 校验、JSON 读取、节点组装与
别名契约、文件级重写与 IO。`ANNCFused` 属性契约以
[tensorflow_addons/annc_fused_op_register.cc](../../../tensorflow_addons/annc_fused_op_register.cc)
为准；Fusion Metadata 字段语义以
[annc/include/FusionMetadata/FusionMetadata.h](../../../annc/include/FusionMetadata/FusionMetadata.h)
为准，镜像结构体不引入新契约。
