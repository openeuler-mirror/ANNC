# ANNC TensorFlow 构建与部署

> 本文是 ANNC TensorFlow predictor 的整体端到端测试手册：addon 构建、predictor 启动、特性使能、e2e 与模型压测。配套脚本位于 `tests/e2e/tf/`。

> 目标：在 aarch64（openEuler + 鲲鹏）上构建 TensorFlow addon
> （`libannc_optimizer.so` + `libannc_fused_op.so`），在 `predictor_server` 中通过
> Grappler 插件启用 ANNC。
> 验证基线：本地 ANNC 工作树（仓库地址
> `https://gitcode.com/openeuler/ANNC.git`）。远程合入后以 `dev` 为准。
> `ANNC_GEMM` 默认关闭，完整 GEMM/epilogue 融合需要显式设置。
> TF 构建不使用 MLAS，不需要 MLAS 头文件或归档。

## 0. 路径约定

| 占位符 | 说明（按实际环境替换） |
|---|---|
| `<ANNC_ROOT>` | ANNC 源码根目录 |
| `<ANNC_PREFIX>` | 安装前缀（如 `/opt/ANNC` 或 `<ANNC_ROOT>/install`） |
| `<TF_ROOT>` | TensorFlow 构建/安装根目录（含 `install/bin/predictor_server`） |
| `<MODEL_ROOT>` | 模型与压测数据根目录 |
| `<BRPC_CLIENT>` | brpc 压测客户端可执行文件 |
| `<GEMM_ASM>` | GEMM-ASM 源码/include 树（含 `Sve/`、`Neon/`） |
| `<GEMM_CONFIG>` | 外部 AArch64 GEMM tuning JSON（仓库不跟踪） |
| `<ORT_ROOT>` | ORT 1.26.0 源码树（仅 `--all` 需要） |
| `<ORT_BUILD>` | 同一次 ORT 构建的 `Release/` 目录（仅 `--all` 需要） |
| `<PYTHON>` | 构建用 Python（启用 Python 绑定时需 pybind11/nanobind） |

## 1. 前置条件

### 1.1 TensorFlow 侧产物

| 产物 | 要求 |
|---|---|
| `predictor_server` | `<TF_ROOT>/install/bin/predictor_server`；必须导出 Grappler 注册表符号（`LD_PRELOAD` 模式） |
| TF headers SDK | `<TF_ROOT>/install/annc-tensorflow-headers/include`；与 `predictor_server` 同源 |
| TF 运行库 | 与 headers、predictor 同一次源码构建，C++11 ABI 一致 |
| SavedModel | `<MODEL_ROOT>/{wd_dcn,bst_mmoe,din_mmoe}/1` |
| 压测 TSV | `<MODEL_ROOT>/<model>/<model>.tsv`，表头 + `id<TAB>model_input` |
| `brpc_client` | `<BRPC_CLIENT>` |

### 1.2 ANNC 源码与依赖

```shell
git clone https://gitcode.com/openeuler/ANNC.git
cd ANNC
git checkout dev
git log -1 --oneline
```

- `<GEMM_ASM>` 必须包含 `Sve/` 与 `Neon/`，作为汇编 include 树，在 AOT/JIT
  编译期汇编生成的 `.S` kernel；它不是预编译 kernel 二进制。
- 代码基线以本地 ANNC 工作树为准；远程仓库
  （`https://gitcode.com/openeuler/ANNC.git`）不包含 `third_party/KDNN`、
  `benchmark/annc-bench` 等本地/外部组件；它们按特性作为前置条件，见 §1.3。
- TF 构建不使用 MLAS，不需要 `mlas.h` 或四个 MLAS 归档。
- 系统依赖：

  ```shell
  yum install cmake ninja-build protobuf-devel protobuf-compiler \
              gcc gcc-c++ python3-devel numactl unzip rpm cpio \
              llvm-toolset-20-llvm llvm-toolset-20-llvm-devel \
              llvm-toolset-20-llvm-libs llvm-toolset-20-llvm-static \
              llvm-toolset-20-llvm-mlir llvm-toolset-20-llvm-mlir-devel \
              llvm-toolset-20-llvm-mlir-static llvm-toolset-20-python3-mlir \
              llvm-toolset-20-clang
  pip install numpy pytest
  # 仅在 --enable-python（Python 绑定）时需要：
  # pip install pybind11 nanobind
  ```

- LLVM/MLIR 来自外部工具链（本机验证 `llvm-toolset-20`；CMake `find_package` 使用
  预装工具链与库，不锁定版本）；nlohmann/json 由 CMake `FetchContent` 下载。构建不再编译 LLVM，
  无需 50GB 磁盘或 30–60 分钟 LLVM 编译。

### 1.3 特性前置条件（本地/外部组件）

远程仓库不跟踪以下组件。按需要的特性准备；缺失时对应特性不可用，但不一定
导致整个构建失败。

| 特性 | 需要的前置 | 缺失时的行为 |
|---|---|---|
| AArch64 GEMM 编译/调优 | `ANNC_GEMM_CONFIG` 指向的 GEMM tuning JSON（通常来自 `benchmark/annc-bench`） | GEMM pass 报 `requires --config-path or ANNC_GEMM_CONFIG for AArch64 GEMM`，AOT/JIT 无法完成 |
| AArch64 GEMM 微内核/小形状 | `third_party/annc-gemm-kernels/libannc_gemm_microkernels.a`（或 `-DANNC_AARCH64_GEMM_KERNEL_LIBRARY=<path>`） | `ANNC_AARCH64_GEMM_KERNELS_AVAILABLE=OFF`；不链接微内核，`GemmSmallShapeMicrokernel` 测试不注册，其余目标可构建；静态 RHS 预打包及 SVE 主机布局校验请求明确报错，补库后需重新配置和构建 |
| AOT/JIT kernel 汇编（生成 `.S`） | `<GEMM_ASM>` 的 `Sve/`、`Neon/` include 树 | `annc` 无法汇编生成的 `.S`，报 `ANNC_GEMM_ASM_SOURCE_DIR must point to ...`；不是预编译 kernel 二进制 |
| KDNN 加速 | `third_party/KDNN`（LOCAL）或网络/Release（RELEASE）/git（REMOTE），并在构建时 `--enable-kdnn-adaptor` | 不启用 KDNN kernel；不要设置 `ANNC_BACKEND=kdnn`，它不能替代构建开关 |
| 离线构建 | 预置 `third_party/json/`；LLVM/MLIR 20 由系统包提供 | FetchContent（json）需要网络或预置源码 |

> `ANNC_GEMM_CONFIG` 是 AArch64 GEMM 编译的硬前置，不是可选调优项。没有
> `annc-bench` 时，需要提供一份兼容的 tuning JSON。
>
> `<GEMM_ASM>` 是汇编源码/include 树，不是 kernel 二进制。预编译微内核二进制是
> `third_party/annc-gemm-kernels/libannc_gemm_microkernels.a`，用于另一条特性路径。
> 使用构建机预热好的缓存或已编译 AOT 产物时，目标机可能不需要源码；冷编译仍然
> 需要 `Sve/`、`Neon/`。

## 2. 构建与安装

### 2.1 生成 TensorFlow protobuf

`build.sh` 首次构建会自动生成 `tf_protos_minimal/gen_code`。直接使用 `cmake` 时先执行：

```shell
cd <ANNC_ROOT>
./tf_protos_minimal.sh -d <ANNC_ROOT>
```

### 2.2 配置

只构建 TensorFlow 前端 + addon：

```shell
cd <ANNC_ROOT>

export ANNC_TENSORFLOW_INCLUDE_DIR=<TF_ROOT>/install/annc-tensorflow-headers/include
export ANNC_TENSORFLOW_PRELOAD=ON
export ANNC_TENSORFLOW_CXX11_ABI=1
export ANNC_GEMM_ASM_SOURCE_DIR=<GEMM_ASM>

./build.sh --no-install-deps \
  --build-type Release --install-prefix <ANNC_PREFIX> --enable-tf
```

- `--enable-tf`：仅 TensorFlow 前端 + addon，自动关闭 ORT 加速栈。
- Python 绑定默认不构建；需要时加 `--enable-python`（同时安装 pybind11/nanobind）。
- GEMM 预打包能力随构建产物一同提供，运行时由 `ANNC_GEMM=3/99` 启用，
  无需额外编译选项；预打包需要 §1.3 的 GEMM 微内核归档。
- `--all`：同时构建 TensorFlow 与 ORT 加速栈；ORT 侧额外需要 ORT 头文件、
  ORT 构建树与 MLAS 头文件目录，见
  [ort-ep-build-and-deploy.md](ort-ep-build-and-deploy.md)。
- `--dry-run`：先确认框架开关。
- TF 构建不使用 MLAS；本文命令不需要 MLAS 头文件/归档。
- KDNN 默认不启用（`ANNC_ENABLE_KDNN_ADAPTOR=OFF`），默认 TF 构建不会下载或
  使用 KDNN。需要 `ANNC_BACKEND=kdnn` 或 KDNN kernel 时，构建必须追加
  `--enable-kdnn-adaptor`，并按 §1.3 准备 KDNN 源。`--kdnn-source` 默认
  `LOCAL`（使用 `third_party/KDNN`，仓库不跟踪）；离线或改用 `RELEASE`/`REMOTE`
  时按 KDNN 文档指定。predictor 的 `--enable_kdnn=false` 只关闭 TF 自身 KDNN
  路径，不能替代 ANNC 的构建开关。
- 需要 AArch64 GEMM 微内核时，把 `libannc_gemm_microkernels.a` 放到
  `third_party/annc-gemm-kernels/`，或设置
  `-DANNC_AARCH64_GEMM_KERNEL_LIBRARY=<path>`；否则该特性为 OFF，对应测试不注册。
- LLVM/MLIR 查找顺序：`-DLLVM_DIR`/`-DMLIR_DIR` → 环境变量 `ANNC_LLVM_BIN` →
  `LLVM_ROOT`/`MLIR_ROOT` → `llvm-config-{22,21,20} --cmakedir`。标准
  `llvm-toolset-20` 安装且 `llvm-config-20` 在 `PATH` 时无需导出；工具链在自定义
  前缀时：

  ```shell
  export ANNC_LLVM_BIN=<LLVM 20 的 bin 目录>
  # 或：export LLVM_DIR=<.../lib64/cmake/llvm>
  #     export MLIR_DIR=<.../lib64/cmake/mlir>
  ```

  运行期 `annc`/`annc-onnx-pipeline` 仍从 `ANNC_LLVM_BIN`（缺省为构建期记录值）
  解析工具；目标机路径与构建机不同时需在运行环境导出。

等价的直接 `cmake` 配置：

```shell
cmake -S <ANNC_ROOT> -B <ANNC_ROOT>/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=<ANNC_PREFIX> \
  -DANNC_ENABLE_TENSORFLOW=ON \
  -DANNC_ENABLE_TENSORFLOW_ADDON=ON \
  -DANNC_ENABLE_ONNX=OFF \
  -DANNC_ENABLE_ONNXRUNTIME_EP=OFF \
  -DANNC_TENSORFLOW_INCLUDE_DIR=<TF_ROOT>/install/annc-tensorflow-headers/include \
  -DANNC_TENSORFLOW_PRELOAD=ON \
  -DANNC_TENSORFLOW_CXX11_ABI=1 \
  -DANNC_GEMM_ASM_SOURCE_DIR=<GEMM_ASM>
```

需要 Python 绑定时（默认不构建），追加 `-DANNC_ENABLE_PYTHON=ON` 与：

```shell
  -Dpybind11_DIR="$(<PYTHON> -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
  -Dnanobind_DIR="$(<PYTHON> -c 'import nanobind; print(nanobind.cmake_dir())')" \
  -DPython_EXECUTABLE="$(command -v <PYTHON>)"
```

### 2.3 构建与安装

`build.sh` 已包含构建与安装；直接 `cmake` 时执行：

```shell
ninja -C <ANNC_ROOT>/build -j16
ninja -C <ANNC_ROOT>/build install
```

`-j16` 按容器 CPU 配额取值（本机 cgroup 配额为 16），不要直接用 `nproc`。

安装前缀关键产物：

```text
<ANNC_PREFIX>/bin/annc-tf-pipeline
<ANNC_PREFIX>/bin/annc-tf2atir
<ANNC_PREFIX>/bin/annc-converter
<ANNC_PREFIX>/bin/annc
<ANNC_PREFIX>/bin/annc-opt
<ANNC_PREFIX>/bin/annc-asm
<ANNC_PREFIX>/lib64/libannc_optimizer.so   # lib/ 布局时为 lib/
<ANNC_PREFIX>/lib64/libannc_fused_op.so   # lib/ 布局时为 lib/
<ANNC_PREFIX>/lib64/libANNC*.a
<ANNC_PREFIX>/lib64/libannc_gemm_microkernels.a   # 仅当外部微内核归档存在时
```

- `ANNC_KERNEL_LIB_PATH` 指向 ANNC 归档目录（安装时由 GNUInstallDirs 决定：
  aarch64 为 `<ANNC_PREFIX>/lib64`，其它布局可能为 `<ANNC_PREFIX>/lib`）。
- LLVM 静态归档（`libLLVMSupport.a`/`libLLVMDemangle.a`）不在 ANNC 安装树，
  由 `ANNC_LLVM_BIN` 的父目录推导（`lib64` 或 `lib`），见下表。

## 3. 特性使能

### 3.1 运行时环境变量

| 变量 | 作用 | 缺省 |
|---|---|---|
| `ANNC_ENABLE` | 启用 ANNCOptimizer | 未设置不启用 |
| `ANNC_GEMM` | GEMM 融合级别：`0`=关闭，`1`=普通 MatMul，`2`=MatMul + epilogue 尾部融合 | `0`（关闭） |
| `ANNC_PIPELINE_PATH` | `annc-tf-pipeline` 路径 | `/usr/local/bin/annc-tf-pipeline` |
| `ANNC_FUSED_OP_PATH` | 写入 GraphDef 的 `libannc_fused_op.so` 路径 | 与 optimizer 同目录 |
| `ANNC_WORK_DIR` | JIT kernel 工作根目录 | 系统临时目录 |
| `ANNC_GEMM_CONFIG` | 外部 AArch64 GEMM 调优 JSON | 无（需外部提供） |
| `ANNC_GEMM_ASM_SOURCE_DIR` | GEMM-ASM 源码/include 树（`Sve/`、`Neon/`），用于汇编生成的 `.S` | `annc` 编译期默认值 |
| `ANNC_KERNEL_LIB_PATH` | ANNC 归档与微内核归档目录 | `<ANNC_PREFIX>/lib64`（GNUInstallDirs；aarch64） |
| `ANNC_LLVM_BIN` | LLVM 20 bin 目录；其父目录的 `lib64`/`lib` 提供 LLVM 静态归档 | 构建期写入的工具目录 |
| `ANNC_BACKEND` | 可选后端名（如 `kdnn`） | `generic` |
| `ANNC_KEEP_TEMPS` | 保留 pipeline 中间文件 | 关闭 |
| `ANNC_VERBOSE` | 打印 pipeline 命令与诊断 | 关闭 |
| `ANNC_TIMEOUT` | pipeline 超时（秒） | 300 |

要点：

- 完整 GEMM/epilogue 融合需要 **`ANNC_GEMM=2`**；未设置时 MatMul/epilogue 不
  融合。`2` 需要 `ANNC_GEMM_ASM_SOURCE_DIR` 可用。
- `ANNC_ENABLE=1` 启用 ANNCOptimizer 及其图重写；`ANNC_GEMM` 是其中的
  GEMM/epilogue 门控，其他融合算子由图重写流程处理。
- 环境变量按进程读取；修改后重启 predictor。

### 3.2 TF predictor 使能 ANNC 的基本操作

```shell
export ANNC_ENABLE=1
export ANNC_VERBOSE=1
export ANNC_GEMM=2                       # 完整 GEMM/epilogue 融合；不设则默认关闭
export ANNC_PIPELINE_PATH=<ANNC_PREFIX>/bin/annc-tf-pipeline
export ANNC_FUSED_OP_PATH=<ANNC_PREFIX>/lib64/libannc_fused_op.so
export ANNC_WORK_DIR=/var/tmp/annc-tf
export ANNC_GEMM_CONFIG=<GEMM_CONFIG>
export ANNC_TIMEOUT=1800
export LD_PRELOAD=<ANNC_PREFIX>/lib64/libannc_optimizer.so   # lib/ 布局时改为 lib/

export TF_NUM_INTEROP_THREADS=16
export TF_NUM_INTRAOP_THREADS=16
export TF_ENABLE_ONEDNN_OPTS=0

<TF_ROOT>/install/bin/predictor_server \
  --enable_kdnn=false \
  --model_path=<MODEL_ROOT>/wd_dcn/1 \
  --thread_num=16 \
  --batch_size=100 \
  --port=9000 \
  --session_config_pbtxt=<ANNC_ROOT>/tensorflow_addons/session_config.pbtxt
```

- `session_config.pbtxt` 必须包含 `optimizers: "ANNCOptimizer"` 与对应
  `custom_optimizers` 参数。
- baseline：不设置 `LD_PRELOAD`，改用
  `<ANNC_ROOT>/tensorflow_addons/baseline_session_config.pbtxt`。
- `--enable_kdnn=false` 关闭 KDNN；需要 KDNN 时另行配置。
- `ANNC_ENABLE` 接受 `1/t/T/y/Y`；`ANNC_TIMEOUT` 默认 300 秒，大模型建议 1800。
- 仅覆盖 `LD_PRELOAD` 插件模式；linked 模式需 `ANNC_TENSORFLOW_PRELOAD=OFF`。
- 没有 `<GEMM_CONFIG>` 时，先设 `ANNC_GEMM=0/1` 做 predictor smoke；完整
  `ANNC_GEMM=2` 需要外部 tuning JSON。

最小 smoke：

```shell
<BRPC_CLIENT> --server=127.0.0.1:9000 \
  --input_data_path=<MODEL_ROOT>/wd_dcn/wd_dcn.tsv \
  --thread_num=1 --warmup_duration_s=0 --test_duration_s=1 \
  --max_qps=1 --max_inflight=1 \
  --request_min_batch_size=1 --request_max_batch_size=1 \
  --timeout_ms=60000
```

基本验收：

1. predictor 日志出现 `ANNCOptimizer initialized`；
2. 出现 `Invoking annc-tf-pipeline graph rewrite` 与
   `ANNCOptimizer graph rewrite completed successfully`；
3. 日志中没有 `compile_failed`，也没有 `Running model failed`；
4. `ANNC_WORK_DIR` 下生成 `kernel.so`。

### 3.3 ANNC_GEMM 特性

- 打开：`ANNC_GEMM=2`（MatMul + epilogue 尾部融合）；`0` 关闭，`1` 只做普通
  MatMul 融合。
- 前置：`ANNC_GEMM_CONFIG` 指向外部 tuning JSON；`ANNC_GEMM_ASM_SOURCE_DIR`
  指向含 `Sve/`、`Neon/` 的 GEMM-ASM 树。
- 本机按 3.2 设置 `ANNC_GEMM=2` 启动 predictor。

证明：

- `ANNC_VERBOSE=1` 日志中出现 GEMM/epilogue 相关的 pipeline 重写；
- `ANNC_WORK_DIR` 下生成 JIT `kernel.so`；
- 同一输入 baseline 与 ANNC 输出一致（`rtol=1e-3, atol=1e-7`）；
- 用 `ANNC_GEMM=0/1` 重启 predictor 做对照，观察 JIT kernel 数和 QPS 变化。

### 3.4 其他融合算子

除 3.3 的 MatMul/epilogue 融合外，ANNC 还会在图重写流程中尝试其他通用图融合。
当前 TF 路径主要包括 DNN embedding 和 KP 系列融合，不由 `ANNC_GEMM` 控制。
下面列出 pattern 与融合算子的对应关系，再说明启用方式和 predictor 通用验收。
本节 predictor 测试以默认图重写流程中的 9 类融合为范围；显式配置融合另见
3.4.4，不计入 predictor 默认测试的验收项。

#### 3.4.1 融合 pattern 与算子

下表列出默认 `OpFusion` 在 TF 路径中注册的 9 类其他融合。`pattern` 对应融合
元数据中的 `fusion.pattern`；融合算子名称对应 `custom.op_name`。
MatMul/epilogue 仍见 3.3，不在此重复；这 9 类不代表 TF 路径支持的全部融合。

| pattern | 融合算子 |
| --- | --- |
| `dnn_embedding_hash_bucket` | `KPFusedDnnEmbeddingWithHashBucket` |
| `kp_embedding_action_id_gather` | `KPFusedEmbeddingActionIdGatherR{r1}x{r2}` |
| `kp_target_behavior_interaction` | `KPFusedTargetBehaviorInteraction` |
| `kp_sparse_dynamic_stitch` | `KPFusedSparseDynamicStitchN{n}` |
| `kp_sparse_segment_reduce` | `KPFusedSparseSegmentReduce{I32,I64}{Sum,Mean}` |
| `kp_embedding_padding` | `KPFusedEmbeddingPadding` |
| `kp_embedding_padding_fast` | `KPFusedEmbeddingPaddingFast` |
| `kp_sparse_reshape` | `KPFusedSparseReshape{I32,I64}[DynLimit]` |
| `kp_sparse_select` | `KPFusedSparseSelect` |

表中花括号和方括号表示名称变体，不是算子名字面量：`r1`、`r2` 是两个输入的
rank；`n` 是 stitch 分支数；`I32` / `I64` 和 `Sum` / `Mean` 按匹配结果选取；
`DynLimit` 是可选后缀。例如实际名称可以是
`KPFusedEmbeddingActionIdGatherR2x2` 或 `KPFusedSparseSegmentReduceI64Mean`。

实际命中取决于模型子图和 kernel 支持情况，并非启用 optimizer 后全部都会命中。

#### 3.4.2 启用图融合

设置 `ANNC_ENABLE=1` 后，TensorFlow 加载 `ANNCOptimizer`，并由 optimizer 调用
`annc-tf-pipeline` 重写计算图。pipeline 中的通用融合流程（包括 OpFusion 等）会随
图重写执行；默认表中的 DNN embedding / KP 融合没有逐类的独立环境变量。

`ANNC_GEMM` 只控制 MatMul/epilogue 融合：设为 `0` 可在验证其他图融合时关闭
GEMM 路径；设为 `1` 或 `2` 时，GEMM 按 3.3 的定义参与执行。改变环境变量后需
重启 predictor 才会生效。沿用 3.2 的 predictor 启动配置，单独验证其他融合时可用：

```shell
export ANNC_ENABLE=1
export ANNC_VERBOSE=1
export ANNC_GEMM=0
export ANNC_WORK_DIR=/var/tmp/annc-tf-other-fusions
# 然后按 3.2 的 predictor_server 命令启动服务
```

上述 predictor 启动方式不会自动加载额外的 JSON 融合配置；需要显式配置的
`KPFusedGather` 不在此测试范围内，其启用与测试方式见 3.4.4。

#### 3.4.3 通用验收

使用与 baseline 相同的模型、输入和请求参数运行 predictor。baseline 不设置
`LD_PRELOAD`，并使用 `tensorflow_addons/baseline_session_config.pbtxt`；ANNC 臂按
3.2 设置 `LD_PRELOAD` 和 `tensorflow_addons/session_config.pbtxt`。两臂分别重启
服务后，用同一输入发起请求并比较全部模型输出。

通用验收应同时确认图重写流程、kernel 产物和模型结果：

1. predictor 日志出现 `ANNCOptimizer initialized`，确认 optimizer 已加载；
2. 日志出现 `Invoking annc-tf-pipeline graph rewrite` 和
   `ANNCOptimizer graph rewrite completed successfully`，确认图重写流程完成；
3. 检查对应的 kernel 产物：JIT 路径在运行时生成 `kernel.so`；AOT 路径由
   pipeline 生成 `annc_generated_kernel.so`。日志中没有 `compile_failed` 或
   `Running model failed`；不能用 JIT kernel 数验收 AOT 融合；
4. 同一输入下 baseline 与 ANNC 输出符合该模型采用的数值容差；
5. 如需评估收益，固定模型、batch、线程数和绑核方式比较两臂 QPS，测试方法见 3.5.1。

**具体 pattern 的命中检查**

按 3.4.2 设置 `ANNC_WORK_DIR` 后，optimizer 会为每次图重写创建独立子目录，并在
`Invoking annc-tf-pipeline graph rewrite` 日志中输出 `work_dir=`。该目录保留
`model_fused_atir.mlir`，可直接检查其中的 `fusion.pattern`，确认本次图重写命中了
哪些融合规则。

```shell
# 替换为本次 predictor 日志中 work_dir= 后的实际路径
PIPELINE_WORK_DIR=/path/to/current/pipeline_work_dir

# 查看该图中命中的 pattern
grep -n 'fusion.pattern' "$PIPELINE_WORK_DIR/model_fused_atir.mlir"

# 例如，单独验证 sparse dynamic stitch
grep -n 'fusion.pattern = "kp_sparse_dynamic_stitch"' \
  "$PIPELINE_WORK_DIR/model_fused_atir.mlir"
```

针对 3.4.1 中的每一类融合，将示例中的 pattern 替换为表中对应名称。检查到目标
`fusion.pattern`，说明该规则已在本次图重写中生成融合函数；没有匹配行时，不能
将该图计为此 pattern 的命中用例。一次 predictor 启动可能重写多个图，应检查与
待测模型和请求相关的各个 `work_dir`。

#### 3.4.4 补充：显式配置融合（不属于 predictor 默认测试）

TF 路径还支持通过 JSON 配置匹配子图、提取融合函数，再由 `FastCodegen` 生成
融合算子的方式。当前已有以下补充项：

| pattern | 融合算子 | 融合内容 |
| --- | --- | --- |
| `kp_fused_gather_external`（示例配置名） | `KPFusedGather` | StridedSlice + 两次 Unique + 两次 Gather |

`kp_fused_gather_external` 来自
`tests/FileCheck/Dialect/Atir/config-fusion-kp-fused-gather.json`，不是默认
`OpFusion` 注册的 pattern。它已有对应的 C++ rewrite 和 kernel 实现，当前通过
显式配置入口接入 TF 流水线。

**启用入口**：直接调用 `annc-tf-pipeline` 并传入 `--fusion-config <JSON 路径>`。
在完成第 2 节构建的环境中，针对符合配置约束的 GraphDef，可使用：

```shell
ANNC_GEMM=0 <ANNC_PREFIX>/bin/annc-tf-pipeline \
  --input_graphdef /path/to/model.pb \
  --output_graphdef /var/tmp/annc-gather/output.pb \
  --work_dir /var/tmp/annc-gather \
  --fusion-config <ANNC_ROOT>/tests/FileCheck/Dialect/Atir/config-fusion-kp-fused-gather.json \
  --output_tensor output:0 \
  --keep_temps --dump-fusion-metadata --verbose
```

替换模型文件、真实输出 tensor 名以及源码和安装路径；多输出时重复传入
`--output_tensor`。此处是直接调用 pipeline 的配置入口；仅设置 `ANNC_ENABLE=1`
启动 predictor，不会加载这份 JSON。

**现有测试入口**：使用已构建源码树中的 TF GraphDef 集成测试：

```shell
cd <ANNC_ROOT>
ANNC_BUILD_DIR=<ANNC_ROOT>/build <PYTHON> -m pytest \
  tests/integration/fusion/test_config_fusion_kp_fused_gather.py -v
```

`ANNC_BUILD_DIR` 指向包含 `bin/annc-tf2atir`、`bin/annc-opt` 和 `bin/annc-asm`
的实际构建目录，`<PYTHON>` 需安装 pytest。测试使用仓库中的 TF GraphDef fixture，
依次执行 TF → ATIR 转换、外部 JSON 配置融合与 `FastCodegen` lowering。

基本验收：

1. `annc-tf2atir`、`annc-opt` 和 `annc-asm` 均执行成功，退出码为 `0`；
2. `model_fused_atir.mlir` 中生成 `fused_kp_fused_gather_external_` 前缀的融合
   函数，并包含 `fusion.pattern = "kp_fused_gather_external"`，确认配置融合命中；
3. `model_fast_codegen.mlir` 中包含 `custom.op_name = "KPFusedGather"`，
   确认融合结果已转换为目标自定义算子。

以上检查由测试脚本自动完成；pytest 汇总结果为 `1 passed`，即该配置融合与
lowering 测试通过。

### 3.5 使能所有特性

同时开启 `ANNC_ENABLE=1` 与 `ANNC_GEMM=2`，并按 3.2 启动 predictor，即为当前
predictor 默认流程的特性全开（默认内置图融合随 `ANNCOptimizer` 启用）。
3.4.4 的显式配置融合不包含在此启动方式中。

#### 3.5.1 性能 A/B

固定 batch=100、client 1 线程、预热 15s、测量 10s、两臂交错 2 轮：

```shell
<BRPC_CLIENT> --server=127.0.0.1:9000 \
  --input_data_path=<MODEL_ROOT>/wd_dcn/wd_dcn.tsv \
  --thread_num=1 --warmup_duration_s=15 --test_duration_s=10 \
  --max_qps=0 --max_inflight=100 \
  --request_min_batch_size=100 --request_max_batch_size=100 \
  --timeout_ms=60000
```

验收必须分层，`brpc` 的 `success=total` 不能单独证明 ANNC 生效：

1. predictor 日志出现 `ANNCOptimizer initialized`；
2. 出现 `Invoking annc-tf-pipeline graph rewrite` 与
   `ANNCOptimizer graph rewrite completed successfully`；
3. 日志中没有 `compile_failed`，也没有 `Running model failed`；
4. `ANNC_WORK_DIR` 下生成 `kernel.so`；
5. 同一输入下 baseline 与 ANNC 输出一致（`rtol=1e-3, atol=1e-7`）；
6. 同机、同绑核、同线程/批次下比较 QPS。

#### 3.5.2 predictor 端到端

`tests/e2e/tf/e2e_tf_predictor.py` 验证分层契约（必须提供外部环境）：

```shell
export ANNC_PREFIX=<ANNC_PREFIX>
export ANNC_SRC=<ANNC_ROOT>
export TF_PREDICTOR=<TF_ROOT>/install/bin/predictor_server
export TF_MODEL=<MODEL_ROOT>/wd_dcn/1
export TF_DATA=<MODEL_ROOT>/wd_dcn/wd_dcn.tsv
export BRPC_CLIENT=<BRPC_CLIENT>
export ANNC_GEMM=2
export ANNC_GEMM_ASM_SOURCE_DIR=<GEMM_ASM>
python3 tests/e2e/tf/e2e_tf_predictor.py
```

> 该脚本内部固定设置
> `ANNC_GEMM_CONFIG=<ANNC_SRC>/benchmark/annc-bench/configs/gemm-hip09-multi.json`，
> 会覆盖外部导出的同名变量。`benchmark/annc-bench/` 不在远程仓库中，是运行该
> e2e 的外部前置；缺失时 AOT/JIT 会在 GEMM pass 失败。没有该目录时，e2e 与
> 完整 `ANNC_GEMM=2` 压测都不可用；可先用 `ANNC_GEMM=0/1` 做 predictor smoke，
> 不要假定 e2e 可跑。

脚本检查：`ANNCOptimizer` 加载、graph rewrite 完成、无 JIT 编译失败、至少生成一个
JIT kernel `.so`。

#### 3.5.3 模型压测

`tests/e2e/tf/test_model_zoo_annc.sh` 用于多模型 A/B，默认配置路径已指向
`tensorflow_addons/session_config.pbtxt` 与 `baseline_session_config.pbtxt`：

```shell
cd <ANNC_ROOT>
# 没有用 --enable-kdnn-adaptor 构建 KDNN 时，去掉 ANNC_BACKEND=kdnn
ANNC_GEMM=2 ANNC_GEMM_CONFIG=<GEMM_CONFIG> ANNC_BACKEND=kdnn START_CPU=0 \
  bash tests/e2e/tf/test_model_zoo_annc.sh wide_and_deep 1 -1 1 1
```

脚本默认的外部路径（`TFSERVER_PATH`、`MODEL_BASE`）需按当前环境覆盖；日志写入
`<ANNC_ROOT>/tensorflow_addons/logs/`。

2026-10-18 参考数据（鲲鹏 950、TF `aad323a1`、batch 100、两轮交错；当时基线未包含
dev 恢复的 `ANNC_GEMM` 门控，复测请显式设置 `ANNC_GEMM=2`）：

| 模型 | max_abs_diff | JIT kernel 数 | baseline QPS | ANNC QPS | ΔQPS |
|---|---:|---:|---:|---:|---:|
| wd_dcn | 0 | 7 | 344.8 | 355.8 | +3.2% |
| bst_mmoe | 3.7e-09 | 19 | 304.2 | 298.2 | −2.0% |
| din_mmoe | 4.7e-10 | 30 | 474.4 | 439.4 | −7.4% |

`bst_mmoe`/`din_mmoe` 在当前固定 batch 下融合收益可能为负，后续按模型画像调整
`ANNC_GEMM` 级别或融合策略。
