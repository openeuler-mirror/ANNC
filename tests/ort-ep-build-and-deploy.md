# ANNC ORT EP 构建与部署

> 本文是 ANNC ORT EP 的整体端到端测试手册：构建、部署、EP 注册与四个融合特性的使能/验证。配套测试位于 `tests/integration/ort_ep/` 与 `tests/unit/ort_ep/`。

> 目标：在 aarch64（openEuler + 鲲鹏）上构建 `libannc_ep.so`，在 ONNX Runtime
> 1.26.0 推理进程中注册为 `AnncExecutionProvider`。
> 验证基线：本地 ANNC 工作树（仓库地址
> `https://gitcode.com/openeuler/ANNC.git`；含 ORT EP 自动启用 MLAS 的改动）。
> 远程合入后以 `dev` 为准。
> MLAS 是 ORT EP 的必需组件，构建 ORT EP 时自动启用，不需要手动设置。

## 0. 路径约定

| 占位符 | 本机当前值（示例） |
|---|---|
| `<ANNC_ROOT>` | `/home/yyf/ANNC` |
| `<ANNC_PREFIX>` | `/opt/ANNC`（或 `<ANNC_ROOT>/install`） |
| `<ORT_ROOT>` | `/home/yyf/onnx/onnxruntime.bak`（ORT 源码树） |
| `<ORT_BUILD_ROOT>` | `/home/yyf/onnx/onnxruntime.bak/build`（ORT `--build_dir`） |
| `<ORT_BUILD>` | `<ORT_BUILD_ROOT>/Linux/Release` |
| `<ONNX_MODEL_DIR>` | `/home/yyf/onnx/model`（`model1~4.onnx`） |
| `<BENCH_SCRIPT>` | `/home/yyf/onnx/onnxruntime_test/onnxruntime_test/onnx_benchmark_all.py` |
| `<PYTHON>` | `/home/yyf/onnx/venv-ort/bin/python3`（需 onnx/onnxruntime） |

## 1. 前置条件

### 1.1 系统与依赖

- openEuler 24.03 aarch64，GCC 12.x，clang 可用。
- 依赖：

  ```shell
  yum install cmake ninja-build protobuf-devel protobuf-compiler \
              gcc gcc-c++ openssl-devel python3-devel
  pip install pybind11 nanobind numpy
  ```

- LLVM/MLIR 21.1.3 与 nlohmann/json 由 CMake `FetchContent` 下载并编译。首次构建
  约 30–60 分钟、约 50GB；离线环境预置 `third_party/llvm/` 与 `third_party/json/`。

### 1.2 ONNX Runtime 1.26.0

ORT 是 ANNC EP 的宿主，必须使用 **1.26.0**、用 `--build_shared_lib` 构建，并保留
构建树。ANNC 需要**同一次** ORT 构建产出的以下产物；头文件与归档不要混用不同
ORT 源码树：

| 用途 | 位置 |
|---|---|
| EP 头文件 | `<ORT_ROOT>/include/onnxruntime/core/session/`（`onnxruntime_c_api.h`、`onnxruntime_ep_c_api.h`） |
| MLAS 头文件 | `<ORT_ROOT>/onnxruntime/core/mlas/inc/mlas.h` |
| 链接归档 | `<ORT_BUILD>/libonnxruntime_common.a`；`<ORT_BUILD>/_deps/pytorch_cpuinfo-build/libcpuinfo.a` |
| MLAS 运行期归档 | `<ORT_BUILD>/libonnxruntime_mlas.a`、`<ORT_BUILD>/_deps/kleidiai-build/libkleidiai.a` |
| 宿主运行库 | 目标机上的 `libonnxruntime.so.1.26.0` |

已验证的 ORT 构建口径（Release、aarch64；`<ORT_BUILD_ROOT>` 是 `--build_dir`，
产物落在 `<ORT_BUILD>`）：

```shell
cd <ORT_ROOT>
./build.sh --build_dir <ORT_BUILD_ROOT> --config Release --build_shared_lib \
  --parallel 16 --skip_tests --compile_no_warning_as_error --skip_submodule_sync \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
  --cmake_extra_defines onnxruntime_USE_SVE=ON \
  --cmake_extra_defines onnxruntime_USE_KLEIDIAI=ON \
  --cmake_extra_defines onnxruntime_DISABLE_CONTRIB_OPS=OFF
```

- `--parallel` 按容器 CPU 配额取值，不要直接用 `nproc`（本机 cgroup 配额为 16）。
- 不要删除 ORT 构建树：`ANNC_EP_ORT_BUILD_TREE` 和 MLAS 运行期归档都依赖它。
- `<ORT_ROOT>` 与 `<ORT_BUILD_ROOT>` 必须来自同一次 ORT 源码构建；换机时按实际
  布局填写，不要复用其他 ORT 版本的头文件或归档。

### 1.3 ANNC 源码

```shell
git clone https://gitcode.com/openeuler/ANNC.git
cd ANNC
git checkout dev
git log -1 --oneline
```

代码基线以本地 ANNC 工作树为准；远程仓库地址为
`https://gitcode.com/openeuler/ANNC.git`，合入后以 `dev` 为准。
本文按本地工作树（含 ORT EP 自动启用 MLAS 的改动）验证。

## 2. 构建与安装

### 2.1 生成 ONNX protobuf

`build.sh` 首次构建会自动生成 `frontends/onnx/protos/gen_code`。直接使用 `cmake`
时先执行：

```shell
cd <ANNC_ROOT>
./frontends/onnx/protos/onnx_protos_minimal.sh -d <ANNC_ROOT>
```

### 2.2 配置

只构建 ORT 加速栈（ONNX 前端 + EP）：

```shell
cd <ANNC_ROOT>

export ANNC_ONNXRUNTIME_INCLUDE_DIR=<ORT_ROOT>/include/onnxruntime/core/session
export ANNC_EP_ORT_BUILD_TREE=<ORT_BUILD>
export ANNC_MLAS_INCLUDE_DIR=<ORT_ROOT>/onnxruntime/core/mlas/inc

./build.sh --no-install-deps \
  --build-type Release --install-prefix <ANNC_PREFIX> --enable-ort
```

- `--enable-ort`：仅构建 ONNX 前端 + EP；`--enable-ort-ep` 是别名。
- `--dry-run`：先确认框架开关，不 configure/build。
- MLAS 是 ORT EP 的必需组件，构建 ORT EP 时自动启用，不需要手动设置。
  构建日志应出现 `Kernel MLAS: ON`。
- 四个 MLAS 运行期归档由 `ANNC_MLAS_LIB_DIR` 或 `ANNC_KERNEL_LIB_PATH` 解析。
- `ANNC_EP_ORT_BUILD_TREE` 填 ORT 构建树的 `Release/` 目录，不是源码树。

等价的直接 `cmake` 配置：

```shell
cmake -S <ANNC_ROOT> -B <ANNC_ROOT>/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=<ANNC_PREFIX> \
  -DANNC_ENABLE_TENSORFLOW=OFF \
  -DANNC_ENABLE_TENSORFLOW_ADDON=OFF \
  -DANNC_ENABLE_ONNX=ON \
  -DANNC_ENABLE_ONNXRUNTIME_EP=ON \
  -DANNC_ONNXRUNTIME_INCLUDE_DIR=<ORT_ROOT>/include/onnxruntime/core/session \
  -DANNC_EP_ORT_BUILD_TREE=<ORT_BUILD> \
  -DANNC_MLAS_INCLUDE_DIR=<ORT_ROOT>/onnxruntime/core/mlas/inc \
  -Dpybind11_DIR="$(<PYTHON> -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
  -Dnanobind_DIR="$(<PYTHON> -c 'import nanobind; print(nanobind.cmake_dir())')" \
  -DPython_EXECUTABLE="$(command -v <PYTHON>)"
```

直接 `cmake` 时同样不需要手动开启 MLAS；只要
`ANNC_ENABLE_ONNXRUNTIME_EP=ON`，CMake 就会自动启用。

### 2.3 构建与安装

`build.sh` 已包含构建与安装；直接 `cmake` 时执行：

```shell
ninja -C <ANNC_ROOT>/build -j16
ninja -C <ANNC_ROOT>/build install
```

`-j16` 按容器 CPU 配额取值（本机 cgroup 配额为 16），不要直接用 `nproc`。

安装前缀关键产物：

```text
<ANNC_PREFIX>/bin/annc-onnx-pipeline
<ANNC_PREFIX>/bin/annc-onnx2atir
<ANNC_PREFIX>/bin/annc
<ANNC_PREFIX>/bin/annc-opt
<ANNC_PREFIX>/bin/annc-asm
<ANNC_PREFIX>/lib64/libannc_ep.so      # lib/ 布局时为 lib/libannc_ep.so
<ANNC_PREFIX>/lib/libANNC*.a
<ANNC_PREFIX>/lib/libLLVM*.a
```

- `libLLVMSupport.a` 等 LLVM 归档与 ANNC 归档必须在同一目录；`ANNC_KERNEL_LIB_PATH`
  指向该目录（本机为 `<ANNC_PREFIX>/lib`）。
- `libannc_ep.so` 由 GNUInstallDirs 决定安装到 `lib` 或 `lib64`，注册时使用实际
  路径；本机为 `<ANNC_PREFIX>/lib64/libannc_ep.so`。
- MLAS 四个归档在运行时解析：优先 `ANNC_MLAS_LIB_DIR`，其次
  `ANNC_KERNEL_LIB_PATH`。构建期只需要 `mlas.h`。

## 3. 特性使能

### 3.1 运行时环境变量

ANNC 安装前缀保持默认相对布局时大多数变量可省略。常用变量：

| 变量 | 作用 | 缺省 |
|---|---|---|
| `ANNC_EP_FUSION_TENSORDOT_MATMUL` | 对应 `tensordot_matmul` pattern；设为 `1` 开启，`0`/未设置关闭 | 关闭 |
| `ANNC_EP_FUSION_LAYER_NORM` | 对应 `layer_norm` pattern；设为 `1` 开启，`0`/未设置关闭 | 关闭 |
| `ANNC_EP_FUSION_MASKED_MUL` | 对应 `masked_mul` pattern；设为 `1` 开启，`0`/未设置关闭 | 关闭 |
| `ANNC_EP_FUSION_BATCHED_MATMUL_ADD` | 对应 `batched_matmul_add` pattern；设为 `1` 开启，`0`/未设置关闭 | 关闭 |
| `ANNC_EP_PIPELINE` | `annc-onnx-pipeline` 绝对路径 | 与 EP 同目录 → `../bin` → `PATH` |
| `ANNC_KERNEL_LIB_PATH` | ANNC/LLVM 归档目录 | `<ANNC_PREFIX>/lib` |
| `ANNC_MLAS_LIB_DIR` | 四个 MLAS 归档目录 | `ANNC_KERNEL_LIB_PATH` |
| `ANNC_EP_CACHE_DIR` | EP 编译缓存目录 | `$XDG_CACHE_HOME/annc-ep` → `$HOME/.cache/annc-ep` |
| `ANNC_CLANG` | clang 路径或 GCC triple 包装脚本 | 构建期检测值 |
| `ANNC_EP_DEBUG=1` | 打印站点申报、编译与缓存诊断 | 关闭 |

要点：

- 四个 `ANNC_EP_FUSION_*` 默认关闭；设为非 `0` 值（如 `1`）开启，未设置或 `0`
  关闭。
- 修改开关后需要新建 session，并会进入 EP 缓存签名。
- `ANNC_EP_DEBUG` 是 presence-only：设置成 `1`、`0` 或空都会开启；生产环境用完
  记得 `unset`。

### 3.2 改写 onnx_benchmark_all.py

`onnx_benchmark_all.py` 本身没有 EP/provider 参数。复制一份并做两处修改：

```shell
cp <BENCH_SCRIPT> /tmp/onnx_benchmark_all_annc.py
```

1. 在 `args = parser.parse_args()` 之后、`sess_opts = ort.SessionOptions()` 之前加入
   （脚本顶部已 `import os`；若你的副本没有，先补上）：

```python
import ctypes
import onnxruntime.capi.onnxruntime_pybind11_state as rtpy

ctypes.CDLL(
    os.path.join(os.path.dirname(ort.__file__), "capi", "libonnxruntime.so.1.26.0"),
    mode=ctypes.RTLD_GLOBAL)
rtpy.register_execution_provider_library(
    "AnncExecutionProvider", "<ANNC_PREFIX>/lib64/libannc_ep.so")  # 绝对路径；lib/ 布局改路径
```

2. 把 `sess = ort.InferenceSession(model, sess_options=sess_opts)` 改为：

```python
sess = ort.InferenceSession(
    model, sess_options=sess_opts,
    providers=["AnncExecutionProvider", "CPUExecutionProvider"])
```

后续小节都使用这份 `/tmp/onnx_benchmark_all_annc.py`。

### 3.3 tensordot_matmul 特性

- 打开：设置 `ANNC_EP_FUSION_TENSORDOT_MATMUL=1`（默认关闭）。
- 对应 pattern：`tensordot_matmul`；融合算子：`FusedTensordotMatmul`。
- 本机 model4 含该 pattern 站点，用 model4 做开启/关闭对照。

```shell
# 开启
ANNC_EP_FUSION_TENSORDOT_MATMUL=1 ANNC_EP_DEBUG=1 \
  <PYTHON> /tmp/onnx_benchmark_all_annc.py \
  --model <ONNX_MODEL_DIR>/model4.onnx --thread_num 2 --max_qps 0 \
  --max_inflight 8 --batch_size 8 --warmup_duration_s 1 --test_duration_s 2 \
  --intra_threads 4 --inter_threads 2 --output_md tdt_on.md

# 关闭对照（不设置或设为 0）
ANNC_EP_FUSION_TENSORDOT_MATMUL=0 ANNC_EP_DEBUG=1 \
  <PYTHON> /tmp/onnx_benchmark_all_annc.py \
  --model <ONNX_MODEL_DIR>/model4.onnx --thread_num 2 --max_qps 0 \
  --max_inflight 8 --batch_size 8 --warmup_duration_s 1 --test_duration_s 2 \
  --intra_threads 4 --inter_threads 2 --output_md tdt_off.md
```

证明：比较两次日志中的 `GetCapability(...): sites=N claimed=N`；开启时该 pattern
的站点被申报，关闭后对应申报减少或消失。

### 3.4 layer_norm 特性

- 打开：设置 `ANNC_EP_FUSION_LAYER_NORM=1`（默认关闭）。
- 对应 pattern：`layer_norm`；融合算子：`LayerNormalization`。
- 本机 model4 含该 pattern 站点。

证明方式与 3.3 相同：用 model4 跑 `=1` 开启与 `=0` 关闭两组，比较
`GetCapability(...): sites/claimed` 的变化。

### 3.5 masked_mul 特性

- 打开：设置 `ANNC_EP_FUSION_MASKED_MUL=1`（默认关闭）。
- 对应 pattern：`masked_mul`；融合算子：`FusedMaskedMul`。
- 本机 model1 的 21 个站点全部由该 pattern 覆盖。

证明方式与 3.3 相同：用 model1 跑 `=1` 开启与 `=0` 关闭两组。本机实测开启时
`GetCapability(...): sites=21 claimed=21`，关闭或未设置时 `sites=0 claimed=0`。

### 3.6 batched_matmul_add 特性

- 打开：设置 `ANNC_EP_FUSION_BATCHED_MATMUL_ADD=1`（默认关闭）。
- 对应 pattern：`batched_matmul_add`；融合算子：`FusedBatchedMatmulAdd`。
- 本机 model3 含该 pattern 站点。

证明方式与 3.3 相同：用 model3 跑 `=1` 开启与 `=0` 关闭两组，比较
`GetCapability(...): sites/claimed` 的变化。其他 pattern（例如通用 `matmul`）
仍可能融合同一子图，因此关闭后站点不一定全部消失。

### 3.7 使能所有特性

四个 `ANNC_EP_FUSION_*` 默认全部关闭；使能所有特性需要把四个开关都设为 `1`。
基线用未修改的脚本（CPU EP），ANNC 臂用 3.2 的补丁副本：

```shell
# 基线：CPU EP
<PYTHON> <BENCH_SCRIPT> \
  --model <ONNX_MODEL_DIR>/model1.onnx --thread_num 5 --max_qps 0 \
  --max_inflight 100 --batch_size 32 --warmup_duration_s 5 --test_duration_s 30 \
  --intra_threads 16 --inter_threads 8 --output_md baseline.md

# ANNC：四个融合开关全部设为 1
ANNC_EP_FUSION_TENSORDOT_MATMUL=1 ANNC_EP_FUSION_LAYER_NORM=1 \
ANNC_EP_FUSION_MASKED_MUL=1 ANNC_EP_FUSION_BATCHED_MATMUL_ADD=1 \
ANNC_EP_DEBUG=1 <PYTHON> /tmp/onnx_benchmark_all_annc.py \
  --model <ONNX_MODEL_DIR>/model1.onnx --thread_num 5 --max_qps 0 \
  --max_inflight 100 --batch_size 32 --warmup_duration_s 5 --test_duration_s 30 \
  --intra_threads 16 --inter_threads 8 --output_md annc.md
```

证明：对比 `actual_qps`、`avg_latency_us`、`p99_latency_us`，并用
`ANNC_EP_DEBUG=1` 日志中的 `sites/claimed` 确认融合已生效（本机 model1 在四个
开关都设为 `1` 时为 21/21；默认全关时为 0/0）。
