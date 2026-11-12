# 环境变量参考

ANNC 的工具与插件通过环境变量控制部分行为。下表汇总面向用户/运维的变量；CMake / 构建脚本内部变量（如 `ANNC_PATCH_FILE`）和已废弃变量未包含在内。

## 通用（所有 ANNC 工具与插件）

| 变量 | 取值 | 默认值 | 说明 |
|------|------|--------|------|
| `ANNC_LOG_LEVEL` | `off` / `error` / `warn` / `info` / `debug`（大小写不敏感） | `error` | ANNC 内部分级日志级别（`annc/include/Support/Log.h`），输出统一走 stderr；例如 `ANNC_LOG_LEVEL=info` 可查看内置 kernel 注册明细，`off` 可完全静默；无法识别的取值回退 `error` 并向 stderr 输出一次性提示 |

## `annc` driver

| 变量 | 取值 | 默认值 | 说明 |
|------|------|--------|------|
| `ANNC_LLVM_BIN` | LLVM/MLIR 20/21 的 bin 目录 | 构建期由 `LLVM_TOOLS_BINARY_DIR` 写入 | 解析 `mlir-opt`/`mlir-translate`/`opt`/`llc`/`clang`/`FileCheck`；未设置时使用构建期默认值，工具缺失则明确报错。工具链 lib 目录（含 `libLLVMSupport.a`/`libLLVMDemangle.a`）由其父目录推导 |
| `ANNC_KERNEL_LIB_PATH` | kernel 静态库目录 | `${CMAKE_INSTALL_FULL_LIBDIR}`（aarch64 为 `<prefix>/lib64`） | 覆盖内置 kernel、线程池归档的搜索目录；LLVM 静态归档不在其中，由 `ANNC_LLVM_BIN` 的父目录解析 |
| `ANNC_EP_ORT_BUILD_TREE` | 目录路径 | 构建期值/未设置 | 宿主 ORT 构建树的 `Release/` 目录；按 `<tree>/` 与 `<tree>/_deps/*-build/` 的布局定位四个 MLAS 归档 `libonnxruntime_mlas.a`、`libkleidiai.a`、`libonnxruntime_common.a`、`libcpuinfo.a`；缺任一归档即拒绝链接 |
| `ANNC_LIBRARY_NAME` | `.so` 文件名 | 自动生成 | 动态测试使用；`annc` 的 step6 会自动设置，手工两步动态驱动流程需自行设为 `./<符号>.so` |

也可对单次编译使用命令行参数 `--kernel-lib-path <dir>`。命令行参数优先于 `ANNC_KERNEL_LIB_PATH`；未指定时使用构建时配置的安装路径。移动 ANNC 安装目录后，需要通过命令行参数或环境变量指定新的 `lib` 目录。

## `annc-opt` / `OpFusion`（GEMM 融合）

| 变量 | 取值 | 默认值 | 说明 |
|------|------|--------|------|
| `ANNC_GEMM` | `0` / `1` / `2` / `3` / `99` | `0` | GEMM 模式：`0`/未设置关闭自动 GEMM fusion，`1`=Basic（plain MatMul，KDNN adaptor 构建下为 AOT KDNN 站点），`2`=Epilogue（追加 epilogue 尾融合，构建时需要完整 GEMM-ASM 二进制包），`3`=Prepack（plain MatMul + 预打包缓存），`99`=All（Epilogue + 预打包缓存）。这是模式编号而非累进等级，`99` 固定保留给 All，未来模式使用未分配的较小编号。预打包与 epilogue 独立；预打包 pass 始终注册，无需额外优化编译开关；依赖归档的路径要求构建时提供 AArch64 微内核库，模式 `1`/`2` 关闭预打包、`3`/`99` 开启，可变权重使用 `1`/`2`。首尾空白允许，空值视为 `0`；非法值向 stderr 提示合法编号并回退 `0`，`ANNC_LOG_LEVEL=info` 可查看解析后的模式。启动前设置，首次读取后固定，修改后需重启；Op 构造时保存结果，只在首次 JIT / 上一次 shape 缓存未命中时使用。该变量同时控制 TF pipeline 自动 AOT 预打包及 packed KDNN 候选，已生成的 AOT 库不会因变量变化而重写。由 `annc-opt` 的 `OpFusion` 与 `annc-tf-pipeline` 读取；`annc` driver 本身不读取该变量 |

`All` 表示同时启用 epilogue 尾融合和 RHS 预打包，不表示合并所有后端的融合策略。KDNN AOT GEMM outlining 仅在模式 `1` 下启用；模式 `2`、`3`、`99` 使用 native GEMM 路径。因此，模式 `3` 的 plain MatMul 不包含模式 `1` 专有的 KDNN AOT 分支。

## `annc-asm` / FastCodegen

| 变量 | 取值 | 默认值 | 说明 |
|------|------|--------|------|
| `ANNC_FAST_CODEGEN_ENABLE_CUSTOM_OPS` | 逗号分隔的 `custom.op_name` 类型 | 空（全部启用） | 全局 allowlist；非空时仅改写列出的类型 |
| `ANNC_FAST_CODEGEN_DISABLE_CUSTOM_OPS` | 逗号分隔的 `custom.op_name` 类型 | 空 | 全局 denylist；始终优先于 allowlist |
| `ANNC_GEMM_CONFIG` | JSON 文件路径 | 空 | AArch64 GEMM 调优配置；AOT 与 JIT 的 GEMM pipeline 均要求提供。使用 JSON v3，`cache.strategy` 与 `kernel.strategy` 分别选择策略，从同级同名对象读取参数；当前支持 `fixed`，未知策略报错。字段示例见 [GEMM codegen 设计](../design/gemm/aarch64-gemm-codegen.md#策略与规划数据) |

环境变量是默认策略。`--atir-fast-codegen=enable-custom-ops=...` 非空时覆盖环境 allowlist；命令行和环境的 denylist 会合并。allowlist 与 denylist 都为空时，**所有已注册的自定义算子类型都可改写**；GEMM 系列（`MatMul`、`MatMulAdd`、`MatMulAddRelu`）的改写 pattern 仅在 `ANNC_ENABLE_KDNN_ADAPTOR=ON` 构建中注册（`MatMulAddReluWithAddOutput` 例外，无条件注册），因此默认构建里对这些类型的 allowlist 设置多数不产生效果；KDNN 构建中还需加 `enable-kdnn=true`（`annc-tf-pipeline` 会自动传入），否则 KDNN-only 的 GEMM kernel 不会被选中。

## `ANNCOptimizer`（Grappler 插件）

这些变量也可通过 `RewriterConfig.CustomGraphOptimizer.parameter_map` 配置，参数名见说明。`parameter_map` 只作用于 Grappler 阶段；运行时 JIT 与 `ANNCFused` 仍直接读取环境变量（`ANNC_FUSED_OP_PATH` 也没有对应的 `parameter_map` 键）。

| 变量 | 取值 | 默认值 | 说明 |
|------|------|--------|------|
| `ANNC_ENABLE` | `1`/`0`/`true`/`false`/`yes`/`no` | 启用 | **只能开启，不能关闭**：`ANNCOptimizer::Init()` 无条件置启用，真值只是重复置启用；关闭须用 `parameter_map` 的 `enabled=false`。环境变量通道额外接受 `yes/no`；`parameter_map` 的 `enabled` 只接受 `true/1` |
| `ANNC_PIPELINE_PATH` | 路径 | `/usr/local/bin/annc-tf-pipeline` | `annc-tf-pipeline` 可执行文件路径；JIT 从其同目录查找 `annc-asm` 和 `annc`（参数名：`pipeline_path` / `annc_pipeline_path`） |
| `ANNC_WORK_DIR` | 目录路径 | 临时目录 | Grappler pipeline 与运行时 JIT 的编译产物工作目录（参数名：`work_dir` / `annc_work_dir`） |
| `ANNC_BACKEND` | `generic` / `kdnn` 等 | `generic` | 后端类型（参数名：`backend`）。当前只被透传为 `annc-tf-pipeline --backend`，而该工具不解析此参数（静默忽略）；实际生效处只有 `ANNCFused` 运行时的 `openblas` 判断，且该 MatMul 直通路径未实现，设置后融合 MatMul 会报 `Unimplemented` |
| `ANNC_VERBOSE` | `1`/`0`/`true`/`false` | `false` | 输出详细日志（参数名：`annc_verbose` / `verbose`） |
| `ANNC_TIMEOUT` | 正整数（秒） | `300` | 图改写工具 `annc-tf-pipeline` 的调用超时（参数名：`timeout_seconds`）；JIT 编译工具 `annc-asm`、`annc` 也使用此环境变量，每次调用单独计时 |
| `PB_NO_FUNC` | pb2stablehlo：跳过 v1 控制流函数化（Switch/Merge→tf.If），图原样进入改写 |
| `PB_NO_FLATTEN` | pb2stablehlo：跳过 tf_executor 结构扁平化（诊断用） |
| `PB_VERBOSE` | pb2stablehlo：dead-Switch 清除的残留引用调试输出 |
| `PB_VERIFY_STAGES` | pb2stablehlo：导入/扁平化/恢复各 stage 后打印模块校验诊断 |
| `PB_DUMP_IMPORT_GRAPH` | pb2stablehlo：把改写后的 GraphDef 序列化转储到指定路径 |
| `PB_DEBUG_IMPORT_INPUTS` | pb2stablehlo：打印导入输入规格统计（feeds/shape 覆盖） |
| `ANNC_NO_OFFICIAL_INFER` | pb2stablehlo：关闭 pb 域官方 shape 富化（grappler GraphProperties），图保持原样；A/B 量测用，此时由 MLIR 侧 rank 规则引擎承担注册 op |
| `ANNC_NO_TF_RANK_INFER` | pb2stablehlo：关闭 MLIR 侧 tf 方言 rank 规则引擎（legalize 前）；A/B 量测用，残差载体将无 rank 来源 |
| `ANNC_NO_UNRANK_WRAP` | pb2stablehlo：关闭 unrank 一元 op 的 getDim/setDim 序列视图包装（legalize 前）；A/B 量测用，关闭后 unranked 实例由 Smuggle 护栏承接 |
| `ANNC_NO_STRIDED_SLICE_DYN` | pb2stablehlo：关闭动态输入 StridedSlice 直译（常量界限归一化 + real_dynamic_slice）；关闭后走残差通道 |
| `ANNC_NO_STRING_TRANSPORT` | pb2stablehlo：关闭 string 载荷数据操作的 encodeString/decodeString 编解码包裹；关闭后 string op 保持残差 |
| `ANNC_NO_STRING_UNWRAP` | pb2stablehlo：关闭 legalize 后的编解码包裹回滚（tf 副本未转换时还原 string 操作数） |
| `ANNC_KEEP_TEMPS` | `1`/`0`/`true`/`false` | `false` | 是否保留 Grappler pipeline 与运行时 JIT 临时文件（参数名：`keep_temp_files`） |
| `ANNC_FUSED_OP_PATH` | `.so` 路径 | 自动推导 | `libannc_fused_op.so` 路径 |
| `ANNC_V2_REWRITE_INPROC` | `1` | 未设置 | 置 `1` 时 Grappler 改写走 v2 进程内路径：`dlopen` 加载 `libannc_graph_rewriter.so` 并调用其 C 入口；任何失败向 stderr 告警并自动回退 v1 pipeline 路径 |
| `ANNC_V2_FUSION_METADATA` | JSON 文件路径 | 无 | v2 进程内改写的 `fusion_metadata.json` 路径（由 `annc-fusion-metadata` 产出）；未设置时 v2 改写失败并回退 |
| `ANNC_V2_SHARED_LIB` | `.so` 路径 | 无 | v2 进程内改写的 AOT kernel 共享库 |
| `ANNC_V2_ATIR_MODULE` | 文件路径 | 无 | v2 进程内改写的 JIT fusion-only ATIR |
| `ANNC_V2_KERNEL_NAME` | 名称 | 无 | v2 进程内改写的 kernel 名覆盖（仅单 fusion 元数据有效） |

TensorFlow Grappler 的普通 GEMM JIT 可按需配置：

```shell
export ANNC_BACKEND=generic
export ANNC_GEMM=1
export ANNC_TIMEOUT=1800
export ANNC_GEMM_CONFIG=/path/to/gemm_tuning.json
```

`ANNC_BACKEND=generic` 目前只是传给 `annc-tf-pipeline`，该工具不解析此参数，
因此它不会选择或强制 generic kernel。`ANNC_GEMM=1` 启用 Basic GEMM fusion；
GEMM pipeline 需要 `ANNC_GEMM_CONFIG` 指向有效调优 JSON。若要使用 JIT RHS
预打包，改用 `ANNC_GEMM=3` 或 `99`，并按需设置 checkpoint 来源：

```shell
export ANNC_GEMM=3
export ANNC_GEMM_CONFIG=/path/to/gemm_tuning.json
export ANNC_DEMO_CHECKPOINT_DIR=/path/to/checkpoint
```

`ANNC_DEMO_CHECKPOINT_DIR` 仅用于预打包候选的首次准入检查；未设置或不匹配时
使用普通 runtime packing。完整模式语义见上方 `ANNC_GEMM` 条目。

## `ANNCFusedOp` 运行时

| 变量 | 取值 | 默认值 | 说明 |
|------|------|--------|------|
| `ANNC_FUSED_PROFILE` | 非空且非 `0` | 关闭 | 开启融合 Op 性能打点（字段含义见[运行时接入](../architecture/runtime-integration.md#运行时性能打点)） |
| `ANNC_FUSED_PROFILE_INTERVAL` | 正整数 | `100000` | 每隔多少次调用输出一次汇总统计 |
| `ANNC_BACKEND` | 字符串 | 空 | 后端选择；当前仅识别 `openblas`，且该 MatMul 直通路径未实现（设置后融合 MatMul 报 `Unimplemented`） |
| `ANNC_DEMO_CHECKPOINT_DIR` | checkpoint 目录 | 空 | 沿用静态 prepack 的权重来源目录；CF 开启时每个 Op 首次初始化读取，核对输入名称、F32 类型和字节数。未设置或未匹配时使用普通 packing；shape 切换不重复读取 |
| `ANNC_JIT_CACHE_MAX_ENTRIES` | 非负整数 | `64` | 进程内 JIT Ready entry 上限；`0` 关闭缓存。首次创建全局缓存时读取并固定；空值使用默认值，非法值告警后回退 `64` |
| `ANNC_ENABLE_JIT_SHAPE_FASTPATH` | `0` 关闭；其他值或未设置开启 | 开启 | 连续相同 shape 时，在当前 `ANNCFusedOp` 上直接复用最近一次 executable 和 binding，跳过全局缓存 key 构造与查找。仅字面值 `0` 关闭；首次读取后进程内固定，需在进程启动前设置。关闭后仍使用进程级 JIT 缓存 |

## ONNX Runtime EP 插件（`libannc_ep.so`）

仅在宿主进程通过 `RegisterExecutionProviderLibrary` 注册 EP 后生效。四个融合开关默认关闭，需显式设为非 `0` 值（如 `1`）开启。

| 变量 | 取值 | 默认值 | 说明 |
|------|------|--------|------|
| `ANNC_EP_PIPELINE` | 可执行文件路径 | 本 `.so` 同目录 → `../bin` → `PATH` | `annc-onnx-pipeline` 路径；设置但文件不存在时直接禁用运行时编译，不回退 |
| `ANNC_EP_CACHE_DIR` | 目录路径 | `$XDG_CACHE_HOME/annc-ep` → `$HOME/.cache/annc-ep` | 运行时编译缓存根 |
| `ANNC_EP_COMPILE` | `0` 关闭 | 启用 | 关闭运行时编译，模型整体回落 CPU |
| `ANNC_EP_COMPILE_TIMEOUT_S` | 正整数（秒） | `600` | 单次编译超时 |
| `ANNC_EP_KEEP_TEMPS` | 非空且非 `0` | 关闭 | 保留编译失败的工作目录 |
| `ANNC_EP_DEBUG` | 存在即开启 | 关闭 | 编译/申报诊断；设 `0` 仍然开启 |
| `ANNC_EP_TIMING` | 存在即开启 | 关闭 | 每节点耗时拆分；设 `0` 仍然开启 |
| `ANNC_EP_NOOP` | 存在即开启 | 关闭 | 只分配输出不计算；设 `0` 仍然开启 |
| `ANNC_EP_FUSION_TENSORDOT_MATMUL` | `1` 开启；`0`/未设置关闭 | 关闭 | 控制 FusedTensordotMatmul 站点申报 |
| `ANNC_EP_FUSION_LAYER_NORM` | `1` 开启；`0`/未设置关闭 | 关闭 | 控制 LayerNormalization 站点申报 |
| `ANNC_EP_FUSION_MASKED_MUL` | `1` 开启；`0`/未设置关闭 | 关闭 | 控制 FusedMaskedMul 站点申报 |
| `ANNC_EP_FUSION_BATCHED_MATMUL_ADD` | `1` 开启；`0`/未设置关闭 | 关闭 | 控制 FusedBatchedMatmulAdd 站点申报 |
| `ANNC_EP_PACKED_B` | 非空且非 `0` 开启（如 `1`）；`0`/空字符串/未设置关闭 | 关闭 | session 创建时为 MLAS 的固定 f32 B 权重打包，适用于 BMM / FTDMM；关闭保留融合并使用 raw-B。修改后须新建 session，详见[部署手册](../../tests/ort-ep-build-and-deploy.md#37-packed-b-特性) |

四个 `ANNC_EP_FUSION_*` 开关默认关闭；设为非 `0` 值（如 `1`）开启，未设置或 `0` 关闭。开关会转发到 `annc-onnx-pipeline` 的编译子进程，`OpFusion` 在编译期跳过对应 matcher，因此不会生成该融合算子；开关进入 EP 缓存指纹。关闭只影响该特定融合算子，其他 pattern（例如通用 `matmul`）仍可能融合同一子图中的节点。

EP 始终使用 ONNX Runtime 的线程池；不提供 `ANNC_EP_SHARDS` / `ANNC_EP_NATIVE_POOL` 开关。原生池代码始终编入 EP，宿主版本匹配时启用，不匹配时回落公开 `KernelContext_ParallelFor` 路径。

EP 插件的构建选项（`--enable-ort`、`ANNC_ONNXRUNTIME_INCLUDE_DIR`、`ANNC_EP_ORT_BUILD_TREE`、`ANNC_KERNEL_MLAS` 等）见[从源码构建](build-from-source.md)。

## 构建与测试

| 变量 | 取值 | 默认值 | 说明 |
|------|------|--------|------|
| `ANNC_TENSORFLOW_PRELOAD` | `ON` / `OFF` | `ON` | TensorFlow addon 预加载模式 |
| `ANNC_TENSORFLOW_INCLUDE_DIR` | TensorFlow include 根目录 | 自动检测 | CMake 配置时显式指定 TF 头文件路径 |
| `ANNC_TENSORFLOW_LIBRARIES` | TensorFlow 库列表 | 空 | `ANNC_TENSORFLOW_PRELOAD=OFF` 时显式指定 TF 库 |
| `ANNC_TENSORFLOW_CXX11_ABI` | `0` / `1` | 自动检测 | TensorFlow 插件的 C++11 ABI，可显式指定 |
| `ANNC_TENSORFLOW_CXX_STANDARD` | C++ 标准编号，如 `14` / `17` | 自动检测 | TensorFlow 插件的 C++ 标准，可显式指定 |
| `CC` / `CXX` | 编译器路径 | `gcc` / `g++` | `build.sh` 使用的 C/C++ 编译器 |
| `PYTHON` | Python 解释器 | `python3` | `build.sh` 使用的 Python |
| `VIRTUAL_ENV` / `CONDA_PREFIX` | 环境标识 | 空 | `build.sh` 据此判断是否在虚拟环境中 |
| `ANNCOPT` / `ANNASM` / `ANNCFUSIONMETADATA` | 可执行文件路径 | `build/bin/...` | `tests/FileCheck/run.sh` 覆盖工具路径 |
| `MLIROPT` / `FILECHECK` | 可执行文件路径 | `$ANNC_LLVM_BIN/...`（未设置时读取 `build/annc_llvm_bin.txt`） | `tests/FileCheck/run.sh` 覆盖工具路径 |
