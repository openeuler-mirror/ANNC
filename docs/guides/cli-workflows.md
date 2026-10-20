# CLI 工作流

ANNC 的工具链围绕 ATIR 组织：前端把框架模型转成 ATIR，`annc-opt` 做图优化，`annc-asm` 与 `annc` 完成 lowering 与代码生成，运行时插件再把编译产物回接到原框架。工具清单见根目录 [README.md](../../README.md#工具矩阵)。

本文给出各工具的典型用法；参数以 `--help` 为准。

## 端到端 GraphDef 重写

`annc-tf-pipeline` 默认编排 `annc-tf2atir`、`annc-opt`、`annc-asm`、`annc` 和 `annc-converter`，适合直接把 TensorFlow GraphDef 重写为包含 `ANNCFused` 自定义 Op 的 GraphDef。`annc-opt` 在 fusion pattern 创建 kernel func 时写入 `annc.execution_mode`：GEMM pattern 默认走 JIT，其他已注册 fusion 默认走 AOT。pipeline 只把 AOT func 编译进共享库，同时保留完整 fusion-only ATIR 供 JIT func 使用。传入 `--fusion-config` 时，`annc-opt` 优先按外部 JSON 匹配，再运行内置 pattern 作为 fallback。传入 `--dump-fusion-metadata` 时额外调用 `annc-fusion-metadata` 保存 JSON。

> **GEMM 融合需要先打开开关**：`ANNC_GEMM` 未设置时默认为 `0`，`OpFusion` 不做 MatMul/GEMM 融合，也不会产生 JIT kernel，且不会报错。需要 GEMM 融合时先 `export ANNC_GEMM=1`（plain MatMul）或 `ANNC_GEMM=2`（追加 epilogue 尾融合）。取值与依赖见[环境变量参考](environment-variables.md)。

```shell
annc-tf-pipeline \
  --input_graphdef input.pb \
  --output_tensor <output_node> \
  --output_graphdef rewritten.pb \
  --work_dir /tmp/annc_work \
  --keep_temps \
  --verbose
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--input_graphdef <path>` | 输入 TensorFlow GraphDef，必填 |
| `--output_tensor <name>` | 保留指定的图输出节点；GraphDef 输入必填（可重复）。SavedModel 输入请先用 `annc-tf2atir` 转换；本工具的 `--input_graphdef` 只接受 GraphDef |
| `--batch_size <n>` | 把所有首维未知（`-1`）的张量首维固定为该值 |
| `--output_graphdef <path>` | 输出 GraphDef，必填 |
| `--shared_lib_path <path>` | 写入 GraphDef 的运行时 `.so` 路径；不传时使用 pipeline 生成的 kernel |
| `--kernel_name <name>` | 覆盖 `ANNCFused` 使用的 kernel 名称 |
| `--fusion-config <path>` | 外部 fusion pattern JSON；示例见 [configs/fusion/](../../configs/fusion/README.md) |
| `--work_dir <dir>` | 中间文件目录 |
| `--keep_temps` / `--keep_temp_files` | 保留中间产物 |
| `--verbose` / `-v` | 打印每一步命令 |

## 分步编译命令

以下命令是手动执行一次 GraphDef 到 `ANNCFused` GraphDef 的**最小序列**。它与 `annc-tf-pipeline` 不等价：pipeline 会在 `ANNC_GEMM=2` 时运行 `--atir-gemm-epilogue-fusion`，按需要追加 KDNN fast-codegen 与常量折叠，并在没有任何 AOT kernel 时跳过 `annc-asm`/`annc`（`--atir-rank-inference` 已包含在下面的第 2 步中）。

```shell
# 1. TensorFlow GraphDef -> ATIR MLIR（GraphDef 输入必须显式指定输出张量；SavedModel 用 serving_default 签名）
annc-tf2atir input.pb --output_tensor <output_node> -o raw.mlir

# 2. ATIR 算子融合（GEMM 融合需先 export ANNC_GEMM=1 或 2）
annc-opt raw.mlir \
  --atir-rank-inference \
  --atir-identity-canonicalize \
  --atir-op-fusion \
  -o fused.mlir

# 3. 提取 ANNCFused 元数据
annc-fusion-metadata fused.mlir \
  -o fusion_metadata.json

# 4. 筛选 AOT kernel，再 lowering 到低层 MLIR
#    启用 GEMM 融合时，GEMM strategy 需要调优配置：
export ANNC_GEMM_CONFIG=<gemm_tuning.json>
annc-opt fused.mlir --atir-prune-func=execution-mode=aot -o aot.mlir
annc-asm aot.mlir \
  --atir-fast-codegen \
  --annc-aarch64-gemm-pipeline \
  -o lowered.mlir

# 5. lowered MLIR -> 共享库
annc lowered.mlir \
  --shared \
  -o kernel.so

# 6. 使用 ANNCFused 重写 GraphDef
#    AOT fusion 需要 --shared_lib_path；JIT fusion（GEMM 家族默认）需要 --atir_module_path
annc-converter --input_graphdef input.pb \
  --output_graphdef rewritten.pb \
  --shared_lib_path "$(pwd)/kernel.so" \
  --atir_module_path "$(pwd)/fused.mlir" \
  --metadata_json fusion_metadata.json
```

> **AOT 与 JIT 的产物要求不同**：`annc-converter` 按 metadata 中每个 fusion 的 `execution_mode` 做前置校验——含 AOT fusion 时必须给 `--shared_lib_path`，含 JIT fusion 时必须给 `--atir_module_path`（运行时要求绝对路径；converter 只校验参数存在，不校验绝对性与文件是否存在，指向未被 prune 的 fusion-only ATIR，即上面第 2 步的 `fused.mlir`），缺失即报错退出。因此第 2 步的 `fused.mlir` 与第 4 步的 `aot.mlir` 都要保留：`kernel.so` 只含 AOT kernel，JIT fusion 的 kernel 在运行时按实际 shape 现场编译。`annc-tf-pipeline` 的做法与此一致（只把 AOT func 编译进共享库，同时保留完整 fusion-only ATIR 供 JIT func 使用）。
>
> `--kernel_name` 只支持单 fusion 的 metadata。

## `annc-tf2atir`

将 TensorFlow GraphDef 直接转换为 ATIR MLIR，不经过 JSON 中间格式。

```shell
# GraphDef 输入：必须至少指定一个 --output_tensor
annc-tf2atir input.pb --output_tensor <output_node> -o model_raw_atir.mlir

# SavedModel 目录或文件：输出取自 serving_default SignatureDef，无需 --output_tensor
annc-tf2atir saved_model_dir -o model_raw_atir.mlir
```

常用参数：

| 参数 | 说明 |
|------|------|
| `-o <file>` | 输出 MLIR 文件 |
| `--output_tensor <name>` | 输出张量节点名，GraphDef 输入必填（可重复） |
| `--batch_size <n>` | 把所有首维未知（`-1`）的张量首维固定为该值；未指定时按图内形状推断 |

## `annc-opt`

`annc-opt` 是注册了 ATIR 方言和 ANNC pass 的 `mlir-opt` 风格工具，用于 ATIR 优化。

```shell
annc-opt model_raw_atir.mlir --atir-identity-canonicalize --atir-op-fusion -o model_fused_atir.mlir
annc-opt input.mlir --atir-op-fusion -o output.bin -emit-bytecode
```

常用 pass：

| Pass | 说明 |
|------|------|
| `--atir-identity-canonicalize` | 清理语义等价的 Identity 透传，供后续融合匹配规范化图形 |
| `--atir-op-fusion` | 算子融合 |
| `--atir-rank-inference` | 推断未知张量秩 |
| `--atir-gemm-epilogue-fusion` | 识别 GEMM epilogue 融合候选 |

## `annc-fusion-metadata`

从融合后的 ATIR 中提取 `ANNCFused` 元数据 JSON，供 `annc-converter` 使用。

```shell
annc-fusion-metadata model_fused_atir.mlir -o fusion_metadata.json
```

当 `--atir-op-fusion` 在同一个计算图中融合出多个独立 kernel 时，`fusion_metadata.json` 使用顶层 `fusions` 数组保存所有融合信息。每个 fusion 保存 `args` / `outputs`（含 `role`、`tf_name`、`shape`、`rank`、`dtype`）、`member_node_names`、ABI 信息与执行模式。`execution_mode`（`aot` / `jit`）与 `template_fingerprint` 是 `annc-converter` 与运行时消费的字段：缺失 `execution_mode` 会默认成 `aot`，导致 JIT fusion 被按 AOT 要求校验；JIT fusion 缺失 `template_fingerprint` 会在反序列化时被 `validateFusionInfo` 直接拒绝（AOT fusion 可留空），运行时也会以 `InvalidArgument` 拒绝。以下示例按真实 `annc-fusion-metadata` 输出裁剪（省略部分 `pattern_attrs`）：

示例：

```json
{
  "fusions": [
    {
      "name": "annc_fused_dense_MatMul",
      "pattern": "matmul",
      "execution_mode": "jit",
      "kernel_name": "fused_matmul_abcd1234",
      "template_fingerprint": "<规范化 kernel func 的 SHA256>",
      "args": [
        {
          "role": "fixed",
          "tf_name": "dense/kernel",
          "shape": [128, 64],
          "rank": 2,
          "dtype": "f32"
        },
        {
          "role": "dynamic",
          "tf_name": "input",
          "shape": [-1, 128],
          "rank": 2,
          "dtype": "f32"
        }
      ],
      "outputs": [
        {
          "role": "output",
          "tf_name": "dense/MatMul",
          "shape": [-1, 64],
          "rank": 2,
          "dtype": "f32"
        }
      ],
      "abi": "mlir_ciface",
      "dynamic_dims": [0],
      "kernel_arg_order": [1, 0, 2],
      "symbolic_signature": "F:0|S:1;?,?",
      "fallback_function": "original_subgraph",
      "member_node_names": [],
      "pattern_attrs": {}
    }
  ]
}
```

## `annc-asm`

`annc-asm` 是注册了 ATIR、转换 pass 和 AArch64 target pass 的 `mlir-opt` 风格工具，用于将 ATIR lowering 到低层 MLIR。

启用 `--annc-aarch64-gemm-pipeline` 时，若图中存在 GEMM anchor，必须提供调优配置：设置 `ANNC_GEMM_CONFIG=<file>`，或使用 `-annc-aarch64-gemm-pipeline='config-path=<file>'`；否则 GEMM strategy 会报错并使 pass 失败。

```shell
export ANNC_GEMM_CONFIG=<gemm_tuning.json>
annc-opt model_fused_atir.mlir --atir-prune-func=execution-mode=aot -o model_aot_atir.mlir
annc-asm model_aot_atir.mlir \
  --atir-fast-codegen \
  --annc-aarch64-gemm-pipeline \
  -o model_lowered.mlir
```

`--atir-fast-codegen` 在 allowlist 与 denylist 都为空时改写**所有已注册**的自定义算子类型。GEMM 系列（`MatMul`、`MatMulAdd`、`MatMulAddRelu`）的改写 pattern 只在 `ANNC_ENABLE_KDNN_ADAPTOR=ON` 的构建中注册（例外：aarch64 原生的 `MatMulAddReluWithAddOutput` 无条件注册）：默认构建里这些类型一般交给 `--annc-aarch64-gemm-pipeline`；KDNN 构建中还需在 `--atir-fast-codegen` 里加 `enable-kdnn=true`（`annc-tf-pipeline` 会自动传入），默认 `enable-kdnn=false` 时 KDNN-only 的 GEMM kernel 不会被选中。可以按最终生成的 `custom.op_name` 使用 allowlist 或 denylist 控制改写；名称区分大小写，denylist 优先级更高：

```shell
export ANNC_GEMM_CONFIG=<gemm_tuning.json>
# KDNN 构建下只允许 MatMulAdd，其他自定义算子保持原 ATIR
annc-asm model_fused_atir.mlir \
  "--atir-fast-codegen=enable-custom-ops=MatMulAdd enable-kdnn=true" \
  --annc-aarch64-gemm-pipeline -o model_lowered.mlir

# 禁用 MatMul，MatMulAdd / MatMulAddRelu 等其他类型仍可改写
annc-asm model_fused_atir.mlir \
  "--atir-fast-codegen=disable-custom-ops=MatMul enable-kdnn=true" \
  --annc-aarch64-gemm-pipeline -o model_lowered.mlir
```

端到端流程或其他不便修改 `annc-asm` 参数的调用方，可设置进程环境变量 `ANNC_FAST_CODEGEN_ENABLE_CUSTOM_OPS` 和 `ANNC_FAST_CODEGEN_DISABLE_CUSTOM_OPS`。它们是逗号分隔的类型列表，会由 `annc-tf-pipeline` 启动的 `annc-asm` 自动继承。详见[环境变量参考](environment-variables.md)。

## `annc`

`annc` 是编译 driver，将 lowered MLIR 编译成可执行文件或共享库。内部流程为 `mlir-opt -> mlir-translate -> opt -> llc -> clang link`。

```shell
annc tests/e2e/annc-driver/demo2.mlir -t tests/e2e/annc-driver/driver_static.c -o app
annc tests/e2e/annc-driver/demo.mlir --shared -o fused_matmul.so
```

静态 driver 的 `extern` 符号必须与输入 MLIR 导出的函数一致（`demo2.mlir` 配 `driver_static.c`，`demo.mlir` 配 `driver_dynamic.c`）。

动态驱动需分两步（`--shared` 与 `-t` 同时使用时 `-o` 会被共享库和测试可执行文件共用）。库文件名必须与 MLIR 导出的 `_mlir_ciface_` 符号同名，并以 `./` 开头引用，否则 `dlopen` 不搜索当前目录、driver 也无法从文件名派生符号：

```shell
annc tests/e2e/annc-driver/demo.mlir --shared -o fused_matmul.so
clang -O3 tests/e2e/annc-driver/driver_dynamic.c \
  -DM=1536 -DK=1344 -DN=1152 -o test_app -ldl -lm
ANNC_LIBRARY_NAME=./fused_matmul.so ./test_app
```

常用参数：

| 参数 | 说明 |
|------|------|
| `-o <file>` | 输出文件；`--shared` 时为共享库路径 |
| `-t <file>` | C 测试 driver，默认 `test.c`；driver 的 `extern` 符号须与输入 MLIR 一致；与 `--shared` 同用时输出路径会冲突，请分步执行 |
| `-v` / `--verbose` | 打印详细命令 |
| `--shared` / `-shared` | 生成 `.so`，否则生成可执行文件 |

`annc` 链接阶段默认调用系统 `clang`，可通过环境变量 `ANNC_CLANG` 覆盖；kernel 归档的搜索目录可用 `--kernel-lib-path <dir>` 或 `ANNC_KERNEL_LIB_PATH` 指定。详见[环境变量参考](environment-variables.md)。

```shell
ANNC_CLANG=/path/to/clang annc model_lowered.mlir --shared -o kernel.so
```

链接前还会自动调用 `annc-model-kernel-export`，把 lowered MLIR 中标记的 AArch64 微内核导出为 `.S` 并汇编进产物。

## `annc-verify`

用于验证生成 kernel 的正确性。

```shell
annc-verify output.bin \
  --atir-op-verify="kpGenLibPath=path/to/generated_kernel.so"
```

## `annc-converter`

根据融合 metadata 重写二进制 TensorFlow GraphDef，将融合子图替换为 `ANNCFused` 节点。独立调用时可传入 `--metadata_json`；省略该参数时，使用位置参数提供融合后的 ATIR 以提取 metadata。

```shell
annc-converter --input_graphdef input.pb \
  --output_graphdef output.pb \
  --shared_lib_path /abs/path/kernel.so \
  --metadata_json fusion_metadata.json
```

GraphDef 输出固定为二进制 protobuf。常用参数：

| 参数 | 说明 |
|------|------|
| `--verbose` | 打印图重写详情 |
| `--input_graphdef <path>` | GraphDef 重写模式输入 |
| `--output_graphdef <path>` | GraphDef 重写模式输出 |
| `--shared_lib_path <path>` | `ANNCFused` 运行时共享库路径 |
| `--atir_module_path <path>` | JIT 模式下 fusion-only ATIR 路径（运行时要求绝对路径，converter 不校验） |
| `--metadata_json <path>` | 可选的 `annc-fusion-metadata` 输出；省略时需提供融合后的 ATIR 位置参数 |
| `--kernel_name <name>` | 覆盖融合 kernel 名称 |

## `annc-onnx-pipeline`

把整个 ONNX 模型编译成「一个模型一个 `.so`」的 AOT kernel 库，并同时产出**站点报告 `sites.json`**：每个融合站点的 kernel 名、实参/输出契约、ABI，以及该站点的**成员节点名单**（`member_node_names`，被融合吞并的图节点）。它供 ONNX Runtime EP 插件（`addons/onnxruntime/`）调用——EP 在 `GetCapability` 内导出整图、调用本工具、按报告精确申报融合区域。接入设计见 [../design/onnx-support/ep-design.md](../design/onnx-support/ep-design.md)。

```shell
annc-onnx-pipeline --model model.onnx --out-dir <dir> [--work-dir <dir>] [--verbose]
annc-onnx-pipeline --toolchain-info     # 打印工具链指纹 JSON（缓存键用）
```

产物：`<dir>/sites.json`、`<dir>/libannc_kernels.so`（零站点时不产 `.so`）。`.so` **只包含 AOT 站点**：站点报告在 prune 之前生成，因此默认构建的 `sites.json` 可能列出 JIT（`matmul` 家族）站点，但这些站点不会进入 `.so`，EP 也会拒绝申报。`ANNC_ENABLE_KDNN_ADAPTOR=ON` 的构建会自动向 FastCodegen 传 `enable-kdnn=true`；再设 `ANNC_GEMM=1` 时，plain `matmul` 以 AOT 站点进入 `.so`（`matmul_add`/`matmul_add_relu` 不在 EP 白名单内）。中间件走 MLIR bytecode（大模型文本 ATIR 可达数百 MB）。退出码 0 表示流程跑完（含零站点），非 0 表示某阶段失败——调用方应回落到 CPU 执行而不是让会话失败。

前置：`annc` 链接阶段从**单一目录** `ANNC_KERNEL_LIB_PATH`（缺省 `<prefix>/lib`）解析全部归档（ANNC kernel 归档 + `libLLVMSupport.a` 等 LLVM 归档），因此该目录必须来自 `ninja install` 的安装树；工具启动时会预检并给出明确报错。同理，`annc` 以裸名调用 `mlir-opt`/`mlir-translate`/`opt`/`llc`/`clang`，本工具会把自身 bin 目录（构建树另加 `_deps/llvm-build/bin`）前置到子进程 `PATH`。

## Model Zoo 端到端脚本

`tests/e2e/tf/test_model_zoo_annc.sh` 在 TensorFlow Model Server 上对比 baseline 与 ANNc 的端到端延迟。常用控制变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `START_CPU` / `SERVER_START_CPU` | `0` | server 起始核 |
| `CLIENT_NUMA_NODE` / `CLIENT_START_CPU` | server node + 1 | client 绑核 |
| `RUN_BOTH` | `1` | 一次调用同时跑 baseline 与 ANNc |
| `LATENCY_THRESHOLD` | `200000` | `perf_analyzer` 延迟阈值（微秒） |
| `WARMUP_REQUESTS` | `50` | warm-up 请求数 |
| `ANNC_PIPELINE_PATH` | 自动推导 | `annc-tf-pipeline` 路径（默认指向本仓库 `install/bin`） |
| `LLVM_PATH` / `TF_PY_SITE_PACKAGES` | 自动推导 / 自动检测 | LLVM 安装目录与 TensorFlow Python 路径 |
| `ANNC_BACKEND` | `generic` | 透传给 ANNC 的后端名 |
| `ANNC_WORK_BASE` | `/tmp/annc_modelzoo_work` | 工作目录根 |
| `TFSERVER_PATH` / `MODEL_BASE` | 内部硬编码默认（使用前必须覆盖） | 服务端与模型根目录；默认值指向内部环境 |
| `CONFIG_FILE` / `BASELINE_CONFIG_FILE` | 本仓库相对路径（自动） | session 配置文件 |
| `ADDONS_DIR` | `install/lib` 或 `build/lib`（自动选择） | `libannc_optimizer.so` 目录 |
| `BASE_PORT` / `ENABLE_PROFILER` / `ANNC_FUSED_OP_PATH` | 脚本默认 | 端口、profiler 与 fused Op 路径 |

脚本会设置 `ANNC_ENABLE=1`、`ANNC_VERBOSE=0`（可用环境变量覆盖）与 `LD_PRELOAD=libannc_optimizer.so`（避免 verbose 日志污染性能数据），并为每个模型自动生成 `ANNC_WORK_DIR`、传入 `ANNC_SAVEDMODEL_PATH`。`TFSERVER_PATH` 与 `MODEL_BASE` 的默认值指向内部环境，外部使用前必须显式覆盖。
