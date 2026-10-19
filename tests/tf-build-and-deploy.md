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

| 占位符 | 本机当前值（示例） |
|---|---|
| `<ANNC_ROOT>` | `/home/yyf/ANNC` |
| `<ANNC_PREFIX>` | `/opt/ANNC`（或 `<ANNC_ROOT>/install`） |
| `<TF_ROOT>` | `/workspace/tensorflow` |
| `<MODEL_ROOT>` | `/workspace/benchmark` |
| `<BRPC_CLIENT>` | `/usr/local/bin/brpc_client` |
| `<GEMM_ASM>` | `/home/yyf/GEMM-ASM` |
| `<GEMM_CONFIG>` | `/home/yyf/ANNC/benchmark/annc-bench/configs/gemm-hip09-multi.json`（外部 JSON，仓库不跟踪） |
| `<ORT_ROOT>` | `/home/yyf/onnx/onnxruntime.bak`（仅 `--all` 需要） |
| `<ORT_BUILD>` | `/home/yyf/onnx/onnxruntime.bak/build/Linux/Release`（仅 `--all` 需要） |
| `<PYTHON>` | 构建用 Python（需 pybind11/nanobind） |

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
              gcc gcc-c++ openssl-devel python3-devel numactl unzip rpm cpio
  pip install pybind11 nanobind numpy pytest
  ```

- LLVM/MLIR 21.1.3 与 nlohmann/json 由 CMake `FetchContent` 下载并编译；首次构建
  约 30–60 分钟、约 50GB。

### 1.3 特性前置条件（本地/外部组件）

远程仓库不跟踪以下组件。按需要的特性准备；缺失时对应特性不可用，但不一定
导致整个构建失败。

| 特性 | 需要的前置 | 缺失时的行为 |
|---|---|---|
| AArch64 GEMM 编译/调优 | `ANNC_GEMM_CONFIG` 指向的 GEMM tuning JSON（通常来自 `benchmark/annc-bench`） | GEMM pass 报 `requires --config-path or ANNC_GEMM_CONFIG for AArch64 GEMM`，AOT/JIT 无法完成 |
| AArch64 GEMM 微内核/小形状 | `third_party/annc-gemm-kernels/libannc_gemm_microkernels.a`（或 `-DANNC_AARCH64_GEMM_KERNEL_LIBRARY=<path>`） | `ANNC_AARCH64_GEMM_KERNELS_AVAILABLE=OFF`；不链接微内核，`GemmSmallShapeMicrokernel` 测试不注册，其余目标可构建 |
| AOT/JIT kernel 汇编（生成 `.S`） | `<GEMM_ASM>` 的 `Sve/`、`Neon/` include 树 | `annc` 无法汇编生成的 `.S`，报 `ANNC_GEMM_ASM_SOURCE_DIR must point to ...`；不是预编译 kernel 二进制 |
| KDNN 加速 | `third_party/KDNN`（LOCAL）或网络/Release（RELEASE）/git（REMOTE），并在构建时 `--enable-kdnn-adaptor` | 不启用 KDNN kernel；不要设置 `ANNC_BACKEND=kdnn`，它不能替代构建开关 |
| 离线构建 | 预置 `third_party/llvm/`、`third_party/json/` | FetchContent 需要网络或预置源码 |

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
  -DANNC_GEMM_ASM_SOURCE_DIR=<GEMM_ASM> \
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
<ANNC_PREFIX>/lib/libannc_optimizer.so   # 或 lib64
<ANNC_PREFIX>/lib/libannc_fused_op.so   # 或 lib64
<ANNC_PREFIX>/lib/libANNC*.a
<ANNC_PREFIX>/lib/libannc_gemm_microkernels.a   # 仅当外部微内核归档存在时
<ANNC_PREFIX>/lib/libLLVM*.a
```

- LLVM 归档与 ANNC 归档必须在同一目录；`ANNC_KERNEL_LIB_PATH` 指向该目录
  （本机为 `<ANNC_PREFIX>/lib`）。若安装布局把归档放到 `lib64`，相应改为
  `<ANNC_PREFIX>/lib64`。

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
| `ANNC_KERNEL_LIB_PATH` | ANNC/LLVM/微内核归档目录 | `<ANNC_PREFIX>/lib` |
| `ANNC_CLANG` | clang 路径或 GCC triple 包装脚本 | 构建期检测值 |
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
export ANNC_FUSED_OP_PATH=<ANNC_PREFIX>/lib/libannc_fused_op.so
export ANNC_WORK_DIR=/var/tmp/annc-tf
export ANNC_GEMM_CONFIG=<GEMM_CONFIG>
export ANNC_TIMEOUT=1800
export LD_PRELOAD=<ANNC_PREFIX>/lib/libannc_optimizer.so   # lib64 布局改为 lib64/

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

- 打开：`ANNC_ENABLE=1` 启用 `ANNCOptimizer`；pipeline 的通用图融合
  （OpFusion 等）随图重写流程执行，没有单独的运行时开关。
- 与 `ANNC_GEMM` 独立：`ANNC_GEMM` 只控制 MatMul/epilogue，其他融合由图重写
  流程处理。

证明：

- `ANNCOptimizer initialized`；
- `Invoking annc-tf-pipeline graph rewrite` 与
  `ANNCOptimizer graph rewrite completed successfully`；
- `ANNC_WORK_DIR` 下生成 `kernel.so`，且日志无 `compile_failed` /
  `Running model failed`；
- baseline（不设置 `LD_PRELOAD`、使用 `baseline_session_config.pbtxt`）与
  ANNC 臂同输入数值一致。

具体融合 pattern 清单后续完善；当前以 predictor 日志、JIT kernel 数和数值/
性能对照作为通用证据。

### 3.5 使能所有特性

同时开启 `ANNC_ENABLE=1` 与 `ANNC_GEMM=2`，并按 3.2 启动 predictor，即为当前
文档覆盖的所有特性全开（其他图融合随 `ANNCOptimizer` 默认启用）。

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
