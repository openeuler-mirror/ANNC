# 环境变量参考

ANNC 的工具与插件通过环境变量控制部分行为。下表汇总面向用户/运维的变量；CMake / 构建脚本内部变量（如 `ANNC_PATCH_FILE`）和已废弃变量未包含在内。

## 通用（所有 ANNC 工具与插件）

| 变量 | 取值 | 默认值 | 说明 |
|------|------|--------|------|
| `ANNC_LOG_LEVEL` | `off` / `error` / `warn` / `info` / `debug`（大小写不敏感） | `error` | ANNC 内部分级日志级别（`annc/include/Support/Log.h`），输出统一走 stderr；例如 `ANNC_LOG_LEVEL=info` 可查看内置 kernel 注册明细，`off` 可完全静默；无法识别的取值回退 `error` 并向 stderr 输出一次性提示 |

## `annc` driver

| 变量 | 取值 | 默认值 | 说明 |
|------|------|--------|------|
| `ANNC_CLANG` | clang 可执行文件路径 | `clang` | 链接阶段使用的 clang 路径 |
| `ANNC_KERNEL_LIB_PATH` | kernel 静态库目录 | `${CMAKE_INSTALL_PREFIX}/lib` | 覆盖内置 kernel、线程池和 LLVM 静态库的搜索目录 |
| `ANNC_MLAS_LIB_DIR` | 目录路径 | 空（回落到 `ANNC_KERNEL_LIB_PATH`） | `ANNC_KERNEL_MLAS=ON` 构建时，解析 MLAS 归档 `libonnxruntime_mlas.a`、`libkleidiai.a`、`libonnxruntime_common.a`、`libcpuinfo.a` 的目录；缺任一归档即拒绝链接，避免产出带未解析符号的 `.so` |
| `ANNC_GEMM_ASM_SOURCE_DIR` | 目录路径 | 构建期 `-DANNC_GEMM_ASM_SOURCE_DIR` | GEMM-ASM checkout 根目录（须含 `Sve/`、`Neon/`）；`annc` 在存在 `model_generated_kernels/*.S` 时用它汇编微内核，未配置且未编译期指定则报错退出 |
| `ANNC_LIBRARY_NAME` | `.so` 文件名 | 自动生成 | 动态测试使用；`annc` 的 step6 会自动设置，手工两步动态驱动流程需自行设为 `./<符号>.so` |

也可对单次编译使用命令行参数 `--kernel-lib-path <dir>`。命令行参数优先于 `ANNC_KERNEL_LIB_PATH`；未指定时使用构建时配置的安装路径。移动 ANNC 安装目录后，需要通过命令行参数或环境变量指定新的 `lib` 目录。

## `annc-opt` / `OpFusion`（GEMM 融合）

| 变量 | 取值 | 默认值 | 说明 |
|------|------|--------|------|
| `ANNC_GEMM` | `0` / `1` / `2` | `0` | MatMul 融合级别：未设置或 `0` 关闭融合，`1` 只融合 plain MatMul，`2` 追加 epilogue 尾融合（需要 `ANNC_GEMM_ASM_SOURCE_DIR`）。由 `annc-opt` 的 `OpFusion` 与 `annc-tf-pipeline` 读取；`annc` driver 本身不读取该变量 |

## `annc-asm` / FastCodegen

| 变量 | 取值 | 默认值 | 说明 |
|------|------|--------|------|
| `ANNC_FAST_CODEGEN_ENABLE_CUSTOM_OPS` | 逗号分隔的 `custom.op_name` 类型 | 空（全部启用） | 全局 allowlist；非空时仅改写列出的类型 |
| `ANNC_FAST_CODEGEN_DISABLE_CUSTOM_OPS` | 逗号分隔的 `custom.op_name` 类型 | 空 | 全局 denylist；始终优先于 allowlist |
| `ANNC_GEMM_CONFIG` | JSON 文件路径 | 空 | AArch64 GEMM 调优配置；AOT 与 JIT 的 GEMM pipeline 均要求提供 |

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
| `ANNC_TIMEOUT` | 正整数（秒） | `300` | pipeline 调用超时时间（参数名：`timeout_seconds`） |
| `ANNC_KEEP_TEMPS` | `1`/`0`/`true`/`false` | `false` | 是否保留 Grappler pipeline 与运行时 JIT 临时文件（参数名：`keep_temp_files`） |
| `ANNC_FUSED_OP_PATH` | `.so` 路径 | 自动推导 | `libannc_fused_op.so` 路径 |

## `ANNCFusedOp` 运行时

| 变量 | 取值 | 默认值 | 说明 |
|------|------|--------|------|
| `ANNC_FUSED_PROFILE` | 非空且非 `0` | 关闭 | 开启融合 Op 性能打点（字段含义见[运行时接入](../architecture/runtime-integration.md#运行时性能打点)） |
| `ANNC_FUSED_PROFILE_INTERVAL` | 正整数 | `100000` | 每隔多少次调用输出一次汇总统计 |
| `ANNC_BACKEND` | 字符串 | 空 | 后端选择；当前仅识别 `openblas`，且该 MatMul 直通路径未实现（设置后融合 MatMul 报 `Unimplemented`） |
| `ANNC_JIT_CACHE_MAX_ENTRIES` | 非负整数 | `64` | 进程内 JIT Ready entry 上限；`0` 关闭缓存 |

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

四个 `ANNC_EP_FUSION_*` 开关默认关闭；设为非 `0` 值（如 `1`）开启，未设置或 `0` 关闭。开关会转发到 `annc-onnx-pipeline` 的编译子进程，`OpFusion` 在编译期跳过对应 matcher，因此不会生成该融合算子；开关进入 EP 缓存指纹。关闭只影响该特定融合算子，其他 pattern（例如通用 `matmul`）仍可能融合同一子图中的节点。

EP 始终使用 ONNX Runtime 的线程池；不提供 `ANNC_EP_SHARDS` / `ANNC_EP_NATIVE_POOL` 开关。原生池代码始终编入 EP，宿主版本匹配时启用，不匹配时回落公开 `KernelContext_ParallelFor` 路径。

EP 插件的构建选项（`--enable-ort`、`ANNC_ONNXRUNTIME_INCLUDE_DIR`、`ANNC_EP_ORT_BUILD_TREE`、`ANNC_KERNEL_MLAS` 等）见[从源码构建](build-from-source.md)。

## 构建与测试

| 变量 | 取值 | 默认值 | 说明 |
|------|------|--------|------|
| `ANNC_TENSORFLOW_PRELOAD` | `ON` / `OFF` | `ON` | TensorFlow addon 预加载模式 |
| `ANNC_TENSORFLOW_INCLUDE_DIR` | TensorFlow include 根目录 | 自动检测 | CMake 配置时显式指定 TF 头文件路径 |
| `ANNC_TENSORFLOW_LIBRARIES` | TensorFlow 库列表 | 空 | `ANNC_TENSORFLOW_PRELOAD=OFF` 时显式指定 TF 库 |
| `ANNC_TENSORFLOW_CXX11_ABI` | `0` / `1` | `1`（仅 `build.sh`） | TensorFlow C++11 ABI；直接使用 CMake 时该 cache 默认为空，需显式指定 `0` 或 `1` |
| `CC` / `CXX` | 编译器路径 | `gcc` / `g++` | `build.sh` 使用的 C/C++ 编译器 |
| `PYTHON` | Python 解释器 | `python3` | `build.sh` 使用的 Python |
| `VIRTUAL_ENV` / `CONDA_PREFIX` | 环境标识 | 空 | `build.sh` 据此判断是否在虚拟环境中 |
| `ANNCOPT` / `ANNASM` / `ANNCFUSIONMETADATA` / `MLIROPT` / `FILECHECK` | 可执行文件路径 | `build/bin/...`（`MLIROPT`、`FILECHECK` 为 `build/_deps/llvm-build/bin/...`） | `tests/FileCheck/run.sh` 覆盖工具路径 |
