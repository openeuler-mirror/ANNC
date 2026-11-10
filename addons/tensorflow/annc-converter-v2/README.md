# annc-converter-v2

`annc-converter` 的并行重实现：把 Fusion Metadata 应用到原始 TensorFlow
GraphDef 上，将被融合子图替换为 `ANNCFused` 自定义 Op 节点。CLI 与
v1（`annc/tools/annc-converter`）基本一致，有意差异：

- **元数据来源仅支持 `--metadata_json`**（`annc-fusion-metadata` 先从
  融合 ATIR 导出 JSON）；v1 的 ATIR 文本直读路径不进入 v2；
- **不可落地的站点整条丢弃**（宁缺不错，stderr 告警）：输出端点不在
  原图、输入无法解析、常量型输入接到非常量节点、控制边指向被替换
  节点、站点间循环接线；全部丢弃时写出原图并返回成功；
- **仅支持 TensorFlow 2.x**（配置期校验；SDK 的 protobuf 头需
  ≥ v3.22 的 `ErrorCollector` 签名）；
- 被替换的输出节点**始终**生成 Identity 别名（v1 在仍有消费者时跳过）；
- 输出固定为 binary GraphDef；输入支持 `.pb`/`.pbtxt`，不做 gzip。

## 与 v1 的差异：TF/LLVM 进程隔离

v1 同时链接 TF protobuf 与 annc 的 LLVM/MLIR。本工具的形态：

- **本工具进程只含 TensorFlow**：TF 头与 `libtensorflow_framework`
  来自宿主 TF 依赖，不链接任何 annc LLVM/MLIR 目标；
- **annc 工具链进程只含 LLVM/MLIR**：跨进程只交换文件
  （`fusion_metadata.json` + GraphDef pb）；metadata 读取为纯 std::
  复刻（`fusion_meta.h`），只消费 `annc-fusion-metadata` 的显式 JSON
  契约，不解析 MLIR。

`annc-tf-pipeline` 当前仍编排 v1；切换时需把 `annc-fusion-metadata`
的 JSON 导出（现为可选 `--dump-fusion-metadata`）作为 v2 前置步骤。

## 构建与运行

构建命令、TF 依赖变量（与 `tensorflow_addons` 同一组，`build.sh` 透传）
见 [从源码构建 · annc-converter-v2](../../../docs/guides/build-from-source.md)。
两点本地约束：

- TF 头的 protobuf 生成代码在非 NDEBUG 下引用 `absl` log 模板实例，而
  `libtensorflow_framework` 以隐藏可见性藏掉了这些实例（链接必然失败）；
  本目录 CMake 已注入 NDEBUG，故 `ANNC_DEBUG=ON`（全局 `-UNDEBUG`）与
  本工具不兼容，配置期报错。
- `libtensorflow_framework.so.2` 带版本号（无开发者符号链接），构建与
  安装的 RPATH 均写入其所在目录；运行环境库路径不同时，重新配置
  `ANNC_TENSORFLOW_LIBRARIES` 或设 `LD_LIBRARY_PATH`。

## 用法

```bash
annc-converter-v2 --metadata_json fusion_metadata.json \
  --input_graphdef input.pb --output_graphdef rewritten.pb \
  [--shared_lib_path /path/libgenerated.so \
   --atir_module_path /path/fusion.mlir] [--kernel_name k0] [--verbose]
```

`--metadata_json` 由 `annc-fusion-metadata` 产出。输入 GraphDef 按
binary 解析、失败后按 text 重试；输出固定 binary。重写后的图保持原图
输出名（被替换节点处生成 Identity 别名）；站点丢弃路径直接输出原始图。

## 验证与契约

单元测试在 `tests/unit/converter_v2/`（GTest/CTest；standalone 注册在
本目录 CMakeLists）：端点解析、metadata 校验、JSON 读取、节点组装与
别名契约、文件级重写与 IO。`ANNCFused` 属性契约以
[tensorflow_addons/annc_fused_op_register.cc](../../../tensorflow_addons/annc_fused_op_register.cc)
为准；Fusion Metadata 字段语义以
[annc/include/FusionMetadata/FusionMetadata.h](../../../annc/include/FusionMetadata/FusionMetadata.h)
为准，镜像结构体不引入新契约。
