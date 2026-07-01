# ANNC-Next

## 项目简介

ANNC-Next 是基于 MLIR 的 AI 编译工具链，面向 openEuler 操作系统，将 TensorFlow GraphDef 编译为 AArch64 原生代码。项目利用 MLIR 多级中间表示的优势，实现从高层 AI 算子到底层机器码的完整编译流程。

## 环境要求

### 硬件与系统

| 项目 | 要求 |
| --- | --- |
| 操作系统 | openEuler-24.03-LTS-SP4 |
| 处理器 | 鲲鹏 920X |
| LLVM 版本 | 21.1.3 |

### 磁盘空间说明

- 完整编译（含 LLVM 从源码编译）需要约 50GB 磁盘空间
- LLVM 源码从 Gitee 镜像下载
- 首次编译 LLVM 需要 30-60 分钟，增量构建仅重新编译变更文件

### 软件依赖安装

```shell
# Python 依赖（也可使用：pip install -r requirements.txt）
pip install pybind11 nanobind

# TensorFlow 2.15（CMake 会在 configure 时自动检测）
pip install tensorflow==2.15.0

# 系统工具
yum install cmake clang
```

> **注意**：部分命令可能需要 `sudo` 权限执行
>
> `build.sh` 会自动检测并安装 `pybind11` 和 `nanobind`；如不需要自动安装，可传入 `--no-install-deps`。TensorFlow 需要预先手动安装。

### LLVM & MLIR + nlohmann/json

LLVM/MLIR 21.1.3 和 nlohmann/json 由 CMake `FetchContent` 在 configure 时自动从 Gitee 镜像下载到 `third_party/llvm/` 和 `third_party/json/`，并作为 ANNC 构建的一部分从源码编译，无需手动克隆或预编译。

> **注意**：LLVM 首次编译需要 30-60 分钟，约 50GB 磁盘空间。增量构建仅重新编译变更文件。

## 构建 ANNC

### 方式一：一键构建（推荐）

LLVM 和 nlohmann/json 自动拉取并编译：

```shell
./build.sh --install-prefix /opt/ANNC --build-type Debug
```

> **build.sh 常用选项：**
> - `--build-type [Debug|Release|RelWithDebInfo]`：设置构建类型（默认 Release）
> - `--install-prefix <path>`：设置安装路径（默认 ./install）
> - `--enable-libcxx`：启用 libc++
> - `--clean`：清理构建目录后重新构建
> - `--no-install-deps`：跳过 pybind11/nanobind 自动安装

### 方式二：手动构建

```shell
# 克隆仓库
git clone https://gitcode.com/openeuler/ANNC.git --branch dev

# 进入目录
cd ANNC

# 创建构建目录
mkdir -p build && cd build

# CMake 配置
cmake -G Ninja .. \
  -DCMAKE_INSTALL_PREFIX=/opt/ANNC \
  -DCMAKE_BUILD_TYPE=Debug

# 编译并安装
ninja && ninja install
```

## 验证安装

```bash
# 设置环境变量
export PATH=/opt/ANNC/bin:$PATH

# 检查工具是否可用
which annc-opt annc-asm

# 运行测试
python3 -m pytest tests/          # 全部测试
python3 -m pytest tests/test_asm.py -v  # 单个测试文件
```

## 架构概览

### 编译流水线

```
TF SavedModel/GraphDef
  ├─→ tf-adaptor → JSON 图描述
  └─→ annc-tf2atir → ATIR MLIR（直接路径，跳过 JSON）

ATIR MLIR
  ├─→ annc-opt (--atir-op-fusion, --atir-block-fusion) → 融合后的 ATIR
  ├─→ annc-fusion-metadata → 提取 ANNCFused 元数据 JSON
  ├─→ annc-asm (--atir-prune-func, --atir-tiling, --atir-unroll, --convert-atir-to-affine)
  │     └─→ lowered MLIR (affine/linalg)
  ├─→ annc → .so / 可执行文件（driver: mlir-opt → mlir-translate → opt → llc → clang link）
  ├─→ annc-verify → kernel 正确性验证
  └─→ annc-converter → TF SavedModel/GraphDef（反向转换 + GraphDef 重写）

annc-tf-pipeline → 端到端编排上述所有步骤
```

### 自定义方言：ATIR（AI Tensor IR）

ATIR 是 ANNC 的核心自定义 MLIR 方言，专为 AI 张量计算设计。

- **源码位置**：`annc/lib/Dialect/Atir/`（实现），`annc/include/Dialect/Atir/`（TableGen TD 定义）
- **核心 Op**：`MatMulOp`、`AddOp`、`ReluOp`、`ConcatOp`、`CustomizeOp`、`ConstantOp`、`VariableOp`、`ForOp`、`IfOp`、`ParallelOp`、`BufferOp`、`LoadOp`
- **优化 Pass**：`OpFusion`、`EltwiseFusion`、`BlockFusion`、`Tiling`、`Unroll`、`PruneFunc`、`LLMCodeGen`、`FastCodegen`、`Initialize`、`Canonicalize`、`SelectLoweringStrategy`、`Distribute`
- **接口**：`ShapeInfer`（形状推导）、`Interpret`（解释执行）
- **验证**：OpVerify（kernel 正确性验证，通过 `kpGenLibPath` / `llmGenLibPath` 指定验证库）

### Lowering 路径

| 路径 | 说明 |
|------|------|
| `Conversion/AtirToLinalg/` | ATIR → Linalg（通用路径） |
| `Conversion/AtirToAffine/` | ATIR → Affine（主要路径） |
| `Target/aarch64/` | AArch64 后端：`MatmulPackAffine`、`VectorReduction`、`VectorCommonParallel`、`CacheParallel`、`CacheReduction`、`AnncOneShotBufferize` |

### 工具矩阵

| 工具 | 功能 |
|------|------|
| `tf-adaptor` | TF SavedModel → JSON 图描述 |
| `annc-tf2atir` | TF GraphDef → ATIR MLIR（直接转换，无需 JSON 中间步骤） |
| `annc-opt` | ATIR 优化（算子融合、块融合等） |
| `annc-fusion-metadata` | 从融合 ATIR 提取 ANNCFused 元数据（JSON 格式） |
| `annc-asm` | ATIR → lowered MLIR（affine/linalg） |
| `annc` | 编译驱动：MLIR → .so / 可执行文件 |
| `annc-verify` | Kernel 正确性验证 |
| `annc-converter` | ATIR → TF SavedModel（反向转换）+ GraphDef 重写 |
| `annc-tf-pipeline` | 端到端编排（`--tf-graphdef-rewrite`） |

### TensorFlow 集成

- `tensorflow_addons/annc_optimizer.cc` — TF Grappler 插件，自动调用 `annc-tf-pipeline` 重写计算图
- `tensorflow_addons/annc_fused_op.cc` — `ANNCFused` 自定义 TF Op，通过 dlopen 加载 .so 调用 `_mlir_ciface_` 接口
- `tensorflow_addons/annc_fused_op_register.cc` / `annc_optimizer_register.cc` — 注册模块

### Python 绑定

- `python/` 使用 nanobind 将 ATIR 方言暴露给 Python
- `python/annc/` 提供：`builder`、`ops`、`types`、`helper`、`enums`、`dialects/atir`

### 目录结构

| 目录 | 内容 |
|------|------|
| `annc/tools/` | CLI 工具（10 个） |
| `annc/lib/Dialect/Atir/` | ATIR 方言：Op 实现、Passes、Interfaces、OpVerify |
| `annc/lib/Conversion/` | ATIR → Affine / ATIR → Linalg lowering |
| `annc/lib/Target/aarch64/` | AArch64 代码生成 |
| `annc/lib/Adaptor/tensorflow/` | TF 模型解析适配器 |
| `annc/lib/Builder/` | MLIR Op 构建器 |
| `annc/lib/Kernel/` | 内置 kernel（`builtin_kernels/matmul_aarch64`）+ 线程池（`threadpool/`） |
| `annc/lib/CAPI/` | C API（Dialect + Passes） |
| `annc/include/` | 头文件（TableGen 定义、Pass 声明、Kernel API） |
| `annc/include/Dialect-c/` | C API 头文件（`Dialects.h`, `Passes.h`） |
| `python/` | Python 绑定 |
| `tensorflow_addons/` | TF Grappler 插件 + ANNCFused Op |
| `tests/` | 测试 |
| `third_party/json/` | nlohmann/json（CMake FetchContent 自动下载，header-only） |
| `third_party/llvm/` | LLVM/MLIR 21.1.3（CMake FetchContent 自动下载，从源码编译） |
| `third_party/tensorflow/` | TF C++ 依赖（CMakeLists.txt，configure 时自动检测 pip TF） |

## 组件使用方法

### 端到端 GraphDef 重写

`annc-tf-pipeline` 会自动编排 `annc-tf2atir`、`annc-opt`、`annc-fusion-metadata`、`annc-asm`、`annc` 和 `annc-converter`，适合直接把 TensorFlow GraphDef 重写为包含 `ANNCFused` 自定义 Op 的 GraphDef。

```shell
annc-tf-pipeline \
  --input_graphdef input.pb \
  --output_graphdef rewritten.pb \
  --work_dir /tmp/annc_work \
  --keep_temps \
  --verbose
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--input_graphdef <path>` | 输入 TensorFlow GraphDef，必填 |
| `--output_graphdef <path>` | 输出 GraphDef，必填 |
| `--shared_lib_path <path>` | 写入 GraphDef 的运行时 `.so` 路径；不传时使用 pipeline 生成的 kernel |
| `--kernel_name <name>` | 覆盖 `ANNCFused` 使用的 kernel 名称 |
| `--work_dir <dir>` | 中间文件目录 |
| `--keep_temps` / `--keep_temp_files` | 保留中间产物 |
| `--verbose` / `-v` | 打印每一步命令 |

### 分步编译命令

以下命令等价于手动执行一次 GraphDef 到 `ANNCFused` GraphDef 的主流程。

```shell
# 1. TensorFlow GraphDef -> ATIR MLIR
annc-tf2atir input.pb -o raw.mlir

# 2. ATIR 算子融合
annc-opt raw.mlir \
  --atir-op-fusion \
  -o fused.mlir

# 3. 提取 ANNCFused 元数据
annc-fusion-metadata fused.mlir \
  -o fusion_metadata.json

# 4. ATIR lowering 到 affine/linalg 等低层 MLIR
annc-asm fused.mlir \
  --atir-prune-func \
  --convert-atir-to-affine \
  -o lowered.mlir

# 5. lowered MLIR -> 共享库
annc lowered.mlir \
  --shared \
  -o kernel.so

# 6. 使用 ANNCFused 重写 GraphDef
annc-converter fused.mlir \
  --tf-graphdef-rewrite \
  --input_graphdef input.pb \
  --output_graphdef rewritten.pb \
  --shared_lib_path "$(pwd)/kernel.so" \
  --metadata_json fusion_metadata.json
```

### `tf-adaptor`

`tf-adaptor` 将 TensorFlow SavedModel 解析为 JSON 图描述。

```shell
tf-adaptor <saved_model_dir> <output.json>
```

参数：

| 参数 | 说明 |
|------|------|
| `<saved_model_dir>` | TensorFlow SavedModel 目录 |
| `<output.json>` | 输出 JSON 文件 |

### `annc-tf2atir`

`annc-tf2atir` 将 TensorFlow GraphDef 直接转换为 ATIR MLIR，不经过 JSON 中间格式。

```shell
annc-tf2atir input.pb -o model_raw_atir.mlir
```

### `annc-opt`

`annc-opt` 是注册了 ATIR 方言和 ANNC pass 的 `mlir-opt` 风格工具，用于 ATIR 优化。

```shell
annc-opt model_raw_atir.mlir --atir-op-fusion -o model_fused_atir.mlir
annc-opt input.mlir --atir-op-fusion -o output.bin -emit-bytecode
```

常用 pass：

| Pass | 说明 |
|------|------|
| `--atir-op-fusion` | 算子融合 |
| `--atir-block-fusion` | 块融合 |
| `--atir-eltwise-fusion` | Elementwise 融合 |
| `--atir-select-lowering-strategy` | 选择 lowering 策略 |

### `annc-fusion-metadata`

`annc-fusion-metadata` 从融合后的 ATIR 中提取 `ANNCFused` 元数据 JSON，供 `annc-converter --tf-graphdef-rewrite` 使用。

```shell
annc-fusion-metadata model_fused_atir.mlir -o fusion_metadata.json
```

当 `--atir-op-fusion` 在同一个计算图中融合出多个独立 MatMul kernel 时，`fusion_metadata.json` 使用顶层 `fusions` 数组保存所有融合信息。数组中每一项都保持单个融合节点的字段结构，包括 `name`、`pattern`、`kernel_name`、`output_tensor`、`original_nodes`、`inputs`、shape、rank 和 ABI 信息。`annc-converter --tf-graphdef-rewrite` 会遍历 `fusions` 数组，为每个 fusion 分别删除原始子图并插入独立的 `ANNCFused` 节点；旧版单 fusion JSON 对象仍可被读取。

示例：

```json
{
  "fusions": [
    {
      "name": "annc_fused_dense_MatMul",
      "pattern": "matmul",
      "kernel_name": "fused_matmul_abcd1234",
      "output_tensor": "dense/MatMul",
      "original_nodes": ["dense/MatMul"],
      "inputs": ["dense/kernel", "input"]
    }
  ]
}
```

### `annc-asm`

`annc-asm` 是注册了 ATIR、转换 pass 和 AArch64 target pass 的 `mlir-opt` 风格工具，用于将 ATIR lowering 到 affine/linalg 等低层 MLIR。

```shell
annc-asm model_fused_atir.mlir \
  --atir-prune-func \
  --convert-atir-to-affine \
  -o model_lowered.mlir
```

更完整的示例：

```shell
annc-asm output.bin \
  --atir-prune-func \
  --atir-block-fusion \
  --atir-tiling \
  --atir-unroll \
  --convert-atir-to-affine \
  -o asm.mlir
```

### `annc`

`annc` 是编译 driver，将 lowered MLIR 编译成可执行文件或共享库。内部流程为 `mlir-opt -> mlir-translate -> opt -> llc -> clang link`。

```shell
annc model_lowered.mlir -o app
annc model_lowered.mlir --shared -o kernel.so
annc model_lowered.mlir -t tests/annc/driver_dynamic.c -o test_app -v --shared
```

常用参数：

| 参数 | 说明 |
|------|------|
| `-o <file>` | 输出文件 |
| `-t <file>` | C 测试 driver，默认 `test.c` |
| `-v` / `--verbose` | 打印详细命令 |
| `--shared` / `-shared` | 生成 `.so`，否则生成可执行文件 |

运行时可通过 `ANNC_CLANG` 覆盖 clang 路径：

```shell
ANNC_CLANG=/path/to/clang annc model_lowered.mlir --shared -o kernel.so
```

### `annc-verify`

`annc-verify` 用于验证生成 kernel 的正确性。

```shell
annc-verify output.bin \
  --atir-op-verify="kpGenLibPath=path/to/generated_kernel.so"
```

验证 LLM CodeGen 生成的 kernel：

```shell
annc-verify output.bin \
  --atir-op-verify="llmGenLibPath=path/to/llm_generated_kernel.so"
```

生成 LLM 专用 TensorFlow Op Kernel：

```shell
annc-asm output.bin --atir-LLM-CodeGen
```

### `annc-converter`

`annc-converter` 支持两种模式：ATIR 反向生成 TensorFlow SavedModel，或基于融合元数据重写 GraphDef。

生成 SavedModel：

```shell
annc-converter input.mlir -o saved_model_dir
annc-converter input.mlir -o saved_model_dir --text_format
```

重写 GraphDef：

```shell
annc-converter model_fused_atir.mlir \
  --tf-graphdef-rewrite \
  --input_graphdef input.pb \
  --output_graphdef output.pb \
  --shared_lib_path /abs/path/kernel.so \
  --metadata_json fusion_metadata.json
```

常用参数：

| 参数 | 说明 |
|------|------|
| `-o <path>` | SavedModel 输出目录，或默认模式输出位置 |
| `--text_format` | 输出 `.pbtxt` 文本格式 |
| `--verbose` | 打印转换详情 |
| `--tf-graphdef-rewrite` | 启用 GraphDef 重写模式 |
| `--input_graphdef <path>` | GraphDef 重写模式输入 |
| `--output_graphdef <path>` | GraphDef 重写模式输出 |
| `--shared_lib_path <path>` | `ANNCFused` 运行时共享库路径 |
| `--metadata_json <path>` | `annc-fusion-metadata` 生成的元数据 |
| `--kernel_name <name>` | 覆盖融合 kernel 名称 |

### TensorFlow 集成

`tensorflow_addons/` 提供 Grappler 插件和 `ANNCFused` 自定义 TensorFlow Op。

| 文件 | 说明 |
|------|------|
| `tensorflow_addons/annc_optimizer.cc` | Grappler 优化器插件，自动调用 `annc-tf-pipeline` 重写计算图 |
| `tensorflow_addons/annc_fused_op.cc` | `ANNCFused` Op，运行时 `dlopen` 加载 `.so` 并调用 `_mlir_ciface_` 接口 |
| `tensorflow_addons/annc_fused_op_register.cc` | 注册 `ANNCFused` Op |
| `tensorflow_addons/annc_optimizer_register.cc` | 注册 Grappler 优化器 |

Serving 场景下，`ANNCOptimizer` 调用 `annc-tf-pipeline` 后会保留 pipeline work_dir 中的产物，尤其是 `annc_generated_kernel.so`。重写后的 GraphDef 会把该 `.so` 路径写入每个 `ANNCFused` 节点的 `shared_lib_path`，运行时 `ANNCFused` 需要通过 `dlopen` 加载它。`annc` driver 生成 object/LLVM IR 的临时目录使用微秒时间、进程号和重试序号组成唯一目录，支持多个 Serving 实例或多个 Grappler 优化任务并发编译，避免不同进程删除彼此的 `step*.ll` 中间文件。

多 fusion 的 GraphDef 重写会为每个 fusion 生成一个独立的 `ANNCFused` 节点。`fusion_metadata.json` 中如果某个 fusion 带有 `dynamic_dims`，`annc-converter` 写入 `ANNCFused.output_shapes` 时会把这些维度保留为 `?`，由 `ANNCFused` 在运行时根据动态输入的实际维度推导输出形状，避免 Serving 请求 batch 与编译样例 batch 不一致时按固定维度分配输出。

MatMul fusion metadata 保持 `ANNCFused` 的运行时输入约定：fixed shape 输入在前，dynamic shape 输入在后。`OpFusion` 会根据 MatMul 的 lhs/rhs 哪个输入承载输出 batch 维来生成 `kernel_arg_order`，保证 GraphDef 输入顺序满足运行时形状推导，同时 `_mlir_ciface_` 调用仍按 kernel 函数签名接收 lhs、rhs、output。

`ANNCFused` 运行时默认只保留错误日志，避免 Serving 并发请求时 INFO 调试日志影响吞吐和日志可读性。`dlopen`、`dlsym`、memref 构造、输出分配、参数顺序和 kernel 调用失败会输出 `LOG(ERROR)`，用于定位服务启动或运行阶段的关键失败原因。Serving 目录中的 `tensorflow_serving_addons` 需要与 `tensorflow_addons` 保持对应修改一致。

`ANNCFused` 通过 `zero_initialize_outputs` 控制输出 Tensor 是否在调用 kernel 前清零。该 attr 默认 `true`，保持旧 GraphDef 和非全量写输出 kernel 的保守行为；运行时对 `fusion_pattern=dnn_embedding_hash_bucket` 自动关闭清零，因为生成的 embedding kernel 会完整覆盖 `[batch, embedding_dim]` 输出，跳过清零可消除 `avg_output_init` 带来的主要包装层开销。该优化不要求 GraphDef 写入新 attr，避免不同 Serving OpDef 版本下 unknown attr 被忽略。

`ANNCFused` 运行时性能打点默认关闭，排查融合 Op 延迟时可通过环境变量开启：

```bash
ANNC_FUSED_PROFILE=1 ANNC_FUSED_PROFILE_INTERVAL=100000 ...
```

开启后每隔 `ANNC_FUSED_PROFILE_INTERVAL` 次 `ANNCFused` 调用按 profile key 汇总输出平均耗时；未设置时默认每 100000 次输出一次，避免高 QPS 测试时 profile 日志本身干扰压测。key 规则如下：

- MatMul 类融合保持兼容格式：`kernel_name_MxKxN`，如果节点带有 `fusion_pattern`，追加 `pattern=<fusion_pattern>`。
- `dnn_embedding_hash_bucket` 使用 embedding 专用格式：`kernel_name_pattern=dnn_embedding_hash_bucket_weight=<weight_shape>_input=<runtime_input_shape>_out=<metadata_output_shape>`，避免把 embedding weight 和 string/id 输入误解释为 MatMul 的 lhs/rhs。
- 其他融合算子使用通用格式：`kernel_name_pattern=<fusion_pattern>_inputs=<runtime_input_shapes>_out=<metadata_output_shapes>`。

日志字段含义如下：

单次调用的耗时采样保存在 `Compute()` 局部变量中，只有汇总统计写入全局表时加锁，避免并发请求时多个线程改写同一个 `OpKernel` 成员导致统计互相覆盖。

| 字段 | 含义 |
|------|------|
| `avg_load_library` | 首次加载或 `.so` 路径变化时 `dlopen`/`dlsym` 的平均耗时，缓存命中时为 0 |
| `avg_threadpool_setup` | 获取 TensorFlow CPU threadpool 并安装 ANNC TLS threadpool 的平均耗时 |
| `avg_input_memref` | 输入 Tensor 构造成 MLIR memref descriptor 的平均耗时 |
| `avg_output_alloc` | TensorFlow 输出 Tensor 分配的平均耗时 |
| `avg_output_init` | 输出 Tensor 清零初始化的平均耗时 |
| `avg_output_memref` | 输出 Tensor 构造成 MLIR memref descriptor 的平均耗时 |
| `avg_arg_order` | 构造 `_mlir_ciface_` 实参顺序的平均耗时 |
| `avg_kernel` | `_mlir_ciface_<kernel_name>` 真正执行生成 kernel 的平均耗时 |
| `avg_total` | `ANNCFusedOp::Compute` 端到端平均耗时 |
| `avg_overhead` | `avg_total - avg_kernel`，用于判断 TensorFlow Op 包装层开销 |

远端增量编译可按 `env.md` 中环境执行：

```bash
cd /annc/ANNC_E2E/ANNC_zch
source env.sh
cd build
ninja && ninja install
```

Serving 侧建议用以下命令对比开启和关闭 ANNC 的延迟，并在开启 ANNC 时打开打点：

```bash
cd /annc/ANNC_E2E/ANNC_zch/tensorflow_serving_addons
ANNC_FUSED_PROFILE=1 ANNC_FUSED_PROFILE_INTERVAL=100000 ANNC_BACKEND=kdnn START_CPU=0 bash test_model_zoo_annc.sh wide_and_deep 1 -1 1 1
ANNC_BACKEND=kdnn START_CPU=0 bash test_model_zoo_annc.sh wide_and_deep 1 -1 1 0
```

### Python 绑定

`python/annc` 使用 nanobind 暴露 ATIR 构建接口。

| 模块 | 说明 |
|------|------|
| `annc.builder` | `MLIRBuilder`，创建 MLIR module/function |
| `annc.helper` | `kp.jit`，从 Python 函数构建 ATIR module |
| `annc.ops` | `matmul`、`add`、`relu`、`constant`、`ret`、`for_loop` 等 ATIR Op 构建函数 |
| `annc.types` | `TensorType`、`ElemType`、`Value` |
| `annc.dialects` | ATIR dialect 注册 |

示例：

```python
import numpy as np

from annc.helper import kp
from annc.ops import matmul
from annc.types import ElemType


@kp.jit
def fused_matmul(inputs, **kwargs):
    lhs = inputs[0]
    rhs = inputs[1]
    cm = inputs[2]
    matmul(ElemType.FP32(), [1024, 1024], lhs, rhs, cm, False)


module = fused_matmul(
    name="demo",
    outputs=["out"],
    inputs=[
        np.random.randn(1024, 1024).astype(np.float32),
        np.random.randn(1024, 1024).astype(np.float32),
        np.zeros((1024, 1024)).astype(np.float32),
        np.random.randn(1024).astype(np.float32),
    ],
)

module.operation.print(large_elements_limit=16)
```

## 常见问题

1. **编译失败**：请确认已安装所有依赖（pybind11、nanobind、tensorflow、cmake）
2. **编译内存不足**：减少并行编译任务数，使用 `ninja -j4` 或 `make -j4` 等方式限制
3. **LLVM 下载缓慢**：Gitee 镜像偶尔不稳定，可重试或手动下载 LLVM 源码到 `third_party/llvm/`

## 相关项目

- [LLVM](https://llvm.org/)
- [MLIR](https://mlir.llvm.org/)
- [openEuler](https://openeuler.org/zh/)
