# ANNC ORT EP 构建与部署

> 本文是 ANNC ORT EP 的整体端到端测试手册：构建、部署、EP 注册、四个融合特性与可选 packed-B 的使能/验证。配套测试位于 `tests/integration/ort_ep/` 与 `tests/unit/ort_ep/`。

> 目标：在 aarch64（openEuler + 鲲鹏）上构建 `libannc_ep.so`，在 ONNX Runtime
> 1.26.0 推理进程中注册为 `AnncExecutionProvider`。
> 验证基线：本地 ANNC 工作树（仓库地址
> `https://gitcode.com/openeuler/ANNC.git`；含 ORT EP 自动启用 MLAS、可选 packed-B 的改动）。
> 远程合入后以 `dev` 为准。
> MLAS 是 ORT EP 的必需组件，构建 ORT EP 时自动启用，不需要手动设置。

## 0. 路径约定

| 占位符 | 说明（按实际环境替换） |
|---|---|
| `<ANNC_ROOT>` | ANNC 源码根目录 |
| `<ANNC_PREFIX>` | 安装前缀（如 `/opt/ANNC` 或 `<ANNC_ROOT>/install`） |
| `<ORT_ROOT>` | ORT 1.26.0 源码树 |
| `<ORT_BUILD_ROOT>` | ORT `--build_dir` 构建根目录 |
| `<ORT_BUILD>` | `<ORT_BUILD_ROOT>/Linux/Release` |
| `<ONNX_MODEL_DIR>` | 四个测试模型（`model1~4.onnx`）所在目录 |
| `<BENCH_SCRIPT>` | ONNX 压测脚本路径 |
| `<PYTHON>` | 带 onnx/onnxruntime 的 Python 解释器 |

## 1. 前置条件

### 1.1 系统与依赖

- openEuler 24.03 aarch64，GCC 12.x，clang 可用。
- 依赖：

  ```shell
  yum install cmake ninja-build protobuf-devel protobuf-compiler \
              gcc gcc-c++ openssl-devel python3-devel numactl \
              llvm-toolset-20-llvm llvm-toolset-20-llvm-devel \
              llvm-toolset-20-llvm-libs llvm-toolset-20-llvm-static \
              llvm-toolset-20-llvm-mlir llvm-toolset-20-llvm-mlir-devel \
              llvm-toolset-20-llvm-mlir-static llvm-toolset-20-python3-mlir \
              llvm-toolset-20-clang
  pip install numpy
  # 仅在 --enable-python（Python 绑定）时需要：
  # pip install pybind11 nanobind
  ```

- LLVM/MLIR 来自外部工具链（本机验证 `llvm-toolset-20`；CMake `find_package` 使用
  预装工具链与库，不锁定版本）；nlohmann/json 由 CMake `FetchContent` 下载。构建不再编译 LLVM，
  无需 50GB 磁盘或 30–60 分钟 LLVM 编译；离线环境只需预置 `third_party/json/`。
  运行期通过 `ANNC_LLVM_BIN`（缺省为构建期记录的工具目录）解析 LLVM 20 工具。

### 1.2 ONNX Runtime 1.26.0

ORT 是 ANNC EP 的宿主，必须使用 **1.26.0**、用 `--build_shared_lib` 构建，并保留
构建树。ANNC 需要**同一次** ORT 构建产出的以下产物；头文件与归档不要混用不同
ORT 源码树：

| 用途 | 位置 |
|---|---|
| EP 头文件 | `<ORT_ROOT>/include/onnxruntime/core/session/`（`onnxruntime_c_api.h`、`onnxruntime_ep_c_api.h`） |
| MLAS 头文件 | `<ORT_ROOT>/onnxruntime/core/mlas/inc/mlas.h` |
| 链接归档 | `<ORT_BUILD>/libonnxruntime_common.a`；`<ORT_BUILD>/_deps/pytorch_cpuinfo-build/libcpuinfo.a` |
| MLAS 构建/运行期归档 | `<ORT_BUILD>/libonnxruntime_mlas.a`、`<ORT_BUILD>/_deps/kleidiai-build/libkleidiai.a` |
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
- Python 绑定默认不构建；需要时加 `--enable-python`（同时安装 pybind11/nanobind）。
- `--dry-run`：先确认框架开关，不 configure/build。
- MLAS 是 ORT EP 的必需组件，构建 ORT EP 时自动启用，不需要手动设置。
  构建日志应出现 `Kernel MLAS: ON`。
- packed-B 只有运行时开关，无需新增 CMake 选项。EP 构建时直接链接宿主的
  MLAS/KleidiAI/common/cpuinfo 归档，以解析私有 PackB 接口，不能依赖宿主 `.so`
  导出它们。即使运行时关闭 packed-B，构建 EP 也必须提供这四个归档。
- ORT 的 MLAS 与 packed-B 路径均不需要 GEMM-ASM。可选
  `libannc_gemm_microkernels.a` 不存在时仍可构建和运行该路径；无需安装 GEMM-ASM。
- 四个 MLAS 运行期归档由 `ANNC_EP_ORT_BUILD_TREE` 定位；该变量必须指向 ORT
  构建树的 `Release/` 目录（不是源码树），driver 按 `<tree>/` 与
  `<tree>/_deps/*-build/` 的布局查找四个归档。
- LLVM/MLIR 查找顺序：`-DLLVM_DIR`/`-DMLIR_DIR` → 环境变量 `ANNC_LLVM_BIN` →
  `LLVM_ROOT`/`MLIR_ROOT` → `llvm-config-{22,21,20} --cmakedir`。标准
  `llvm-toolset-20` 安装且 `llvm-config-20` 在 `PATH` 时无需导出；工具链在自定义
  前缀时：

  ```shell
  export ANNC_LLVM_BIN=<LLVM 20 的 bin 目录>
  # 或：export LLVM_DIR=<.../lib64/cmake/llvm>
  #     export MLIR_DIR=<.../lib64/cmake/mlir>
  ```

  运行期 `annc-onnx-pipeline`/`annc` 仍从 `ANNC_LLVM_BIN`（缺省为构建期记录值）
  解析工具；目标机路径与构建机不同时需在运行环境导出。

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
  -DANNC_MLAS_INCLUDE_DIR=<ORT_ROOT>/onnxruntime/core/mlas/inc
```

需要 Python 绑定时（默认不构建），追加 `-DANNC_ENABLE_PYTHON=ON` 与：

```shell
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
<ANNC_PREFIX>/lib64/libANNC*.a      # lib/ 布局时为 lib/libANNC*.a
```

- `ANNC_KERNEL_LIB_PATH` 指向 ANNC 归档目录（aarch64 安装为 `<ANNC_PREFIX>/lib64`）。
- `libannc_ep.so` 由 GNUInstallDirs 决定安装到 `lib` 或 `lib64`，注册时使用实际
  路径；本机为 `<ANNC_PREFIX>/lib64/libannc_ep.so`。
- MLAS 四个归档在 EP 构建期用于链接插件，运行时由 `ANNC_EP_ORT_BUILD_TREE`
  定位以链接生成的 kernel；构建期同时需要 `mlas.h`。
- 更新 packed-B 特性时同步部署 `libannc_ep.so`、pipeline/编译工具与 kernel
  归档；执行上下文已扩展为 ABI v4，不能只替换插件并沿用旧 kernel。
  新插件的工具链指纹会使旧编译缓存失效，首次 session 会重新编译。

## 3. 特性使能

### 3.1 运行时环境变量

ANNC 安装前缀保持默认相对布局时大多数变量可省略。常用变量：

| 变量 | 作用 | 缺省 |
|---|---|---|
| `ANNC_EP_FUSION_TENSORDOT_MATMUL` | 对应 `tensordot_matmul` pattern；设为 `1` 开启，`0`/未设置关闭 | 关闭 |
| `ANNC_EP_FUSION_LAYER_NORM` | 对应 `layer_norm` pattern；设为 `1` 开启，`0`/未设置关闭 | 关闭 |
| `ANNC_EP_FUSION_MASKED_MUL` | 对应 `masked_mul` pattern；设为 `1` 开启，`0`/未设置关闭 | 关闭 |
| `ANNC_EP_FUSION_BATCHED_MATMUL_ADD` | 对应 `batched_matmul_add` pattern；设为 `1` 开启，`0`/未设置关闭 | 关闭 |
| `ANNC_EP_PACKED_B` | 固定 f32 B 权重的 MLAS 打包；非空且非 `0`（如 `1`）开启，`0`/空字符串/未设置关闭 | 关闭 |
| `ANNC_EP_PIPELINE` | `annc-onnx-pipeline` 绝对路径 | 与 EP 同目录 → `../bin` → `PATH` |
| `ANNC_KERNEL_LIB_PATH` | ANNC 归档目录 | `<ANNC_PREFIX>/lib64`（GNUInstallDirs；aarch64） |
| `ANNC_LLVM_BIN` | LLVM  bin 目录；其父目录的 `lib64`/`lib` 提供 LLVM 静态归档 | 构建期写入的工具目录 |
| `ANNC_EP_ORT_BUILD_TREE` | ORT 构建树 `Release/`；用于定位四个 MLAS 归档 | 构建期值/未设置 |
| `ANNC_EP_CACHE_DIR` | EP 编译缓存目录 | `$XDG_CACHE_HOME/annc-ep` → `$HOME/.cache/annc-ep` |
| `ANNC_EP_DEBUG=1` | 打印站点申报、编译与缓存诊断 | 关闭 |

要点：

- 四个 `ANNC_EP_FUSION_*` 默认关闭；设为非 `0` 值（如 `1`）开启，未设置或 `0`
  关闭。
- 修改融合开关后需要新建 session，并会进入 EP 缓存签名。
- packed-B 独立于融合开关，修改后需要新建 session；只改变运行时权重表示，
  不改变融合边界，无需为 `0/1` 分开清理编译缓存。
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
- 本机 model4 含该 pattern 站点。

测试命令与四模型 A/B 同参；`numactl -C 48-63` 按目标机 cpuset 调整（本机实测
用 48-63）：

```shell
# 开启
ANNC_EP_FUSION_TENSORDOT_MATMUL=1 ANNC_EP_DEBUG=1 \
  numactl -C 48-63 <PYTHON> /tmp/onnx_benchmark_all_annc.py \
  --model <ONNX_MODEL_DIR>/model4.onnx --thread_num 5 --max_qps 0 \
  --max_inflight 100 --batch_size 32 --warmup_duration_s 5 \
  --test_duration_s 30 --intra_threads 16 --inter_threads 8 \
  --timeout_ms 30000 --output_md tdt_on.md --debug 2>&1 | tee tdt_on.log

# 关闭对照（不设置或设为 0）
ANNC_EP_FUSION_TENSORDOT_MATMUL=0 ANNC_EP_DEBUG=1 \
  numactl -C 48-63 <PYTHON> /tmp/onnx_benchmark_all_annc.py \
  --model <ONNX_MODEL_DIR>/model4.onnx --thread_num 5 --max_qps 0 \
  --max_inflight 100 --batch_size 32 --warmup_duration_s 5 \
  --test_duration_s 30 --intra_threads 16 --inter_threads 8 \
  --timeout_ms 30000 --output_md tdt_off.md --debug 2>&1 | tee tdt_off.log
```

基线（CPU EP）用未修改的 `<BENCH_SCRIPT>`，不带 `ANNC_EP_FUSION_*` 和
`ANNC_EP_DEBUG`，其余参数相同。

**日志在哪看**：EP 和 pipeline 的诊断都输出到进程 **stderr**；`--output_md`
只写压测报告，不包含站点信息。上面的 `2>&1 | tee` 会把两者都写进日志，然后：

```shell
grep -E 'GetCapability|site\(s\)' tdt_on.log tdt_off.log
```

正常情况下能看到两类行：

```text
[annc-ep] GetCapability(runtime-compile): nodes=<N> sites=<S> claimed=<C> dropped=<D> ...
[annc-onnx-pipeline] <S> site(s) -> <cache>/libannc_kernels.so
```

- `[annc-onnx-pipeline] N site(s)`：运行时编译成功并生成了含 N 个站点的
  `libannc_kernels.so`。这一行是 pipeline 在编译结束时打印的，较稳定。
- `[annc-ep] GetCapability(...)`：EP 在创建 session 时打印的站点申报摘要；
  只有当前 EP 版本带该诊断且 `ANNC_EP_DEBUG` 在 EP 进程里生效时才会出现。
  如果只看到 pipeline 的 `N site(s)`、没有 `GetCapability`，以 pipeline 的
  N 为准，并检查 `ANNC_EP_DEBUG` 是否设置在加载 EP 的进程里、EP 是否为旧版本。
  可用 `strings <ANNC_PREFIX>/lib64/libannc_ep.so | grep 'GetCapability(runtime-compile)'`
  确认 EP 二进制是否包含该诊断。
- 关闭组如果没有 `site(s)` 行，说明编译了 0 个站点，等价于 `sites=0`。

判定：开启时目标 pattern 的站点数应等于预期值（见下表），`claimed=sites`、
`dropped=0`；关闭时对应站点为 0。若 `claimed` 小于 `sites` 或 `dropped>0`，
在日志里搜 `dropping site` 查看被丢弃的站点及原因。

model4 只开 `tensordot_matmul` 时预期 `sites=15 claimed=15 dropped=0`；`=0` 时
预期 `sites=0 claimed=0`。

导入阶段出现以下 warning 是正常的：

```text
warning: unsupported op 'Gemm' at node '...' lowered to atir.opaque
warning: unsupported op 'Slice' at node '...' lowered to atir.opaque
```

**各模型/特性的站点数**（单独开启对应开关、其余保持默认关闭）：

| 模型 | 特性 | 开启时 `sites/claimed` | 关闭时 |
|---|---|---|---|
| model1 | `masked_mul` | 21/21 | 0/0 |
| model2 | `masked_mul` | 33/33 | 0/0 |
| model3 | `masked_mul` | 3/3 | 0/0 |
| model3 | `batched_matmul_add` | 50/50 | 0/0 |
| model4 | `layer_norm` | 27/27 | 0/0 |
| model4 | `tensordot_matmul` | 15/15 | 0/0 |

四个开关全开时，`GetCapability` 应为 model1 21/21、model2 33/33、model3 53/53
（`batched_matmul_add` 50 + `masked_mul` 3）、model4 42/42（`layer_norm` 27 +
`tensordot_matmul` 15）。

### 3.4 layer_norm 特性

- 打开：设置 `ANNC_EP_FUSION_LAYER_NORM=1`（默认关闭）。
- 对应 pattern：`layer_norm`；融合算子：`LayerNormalization`。
- 本机 model4 含该 pattern 站点。

命令同 3.3，把开关换成 `ANNC_EP_FUSION_LAYER_NORM=1/0`。model4 预期
`sites=27 claimed=27`；关闭后 `sites=0 claimed=0`。

### 3.5 masked_mul 特性

- 打开：设置 `ANNC_EP_FUSION_MASKED_MUL=1`（默认关闭）。
- 对应 pattern：`masked_mul`；融合算子：`FusedMaskedMul`。
- 本机 model1/2/3 含该 pattern 站点。

命令同 3.3，把开关换成 `ANNC_EP_FUSION_MASKED_MUL=1/0`，分别跑 model1、
model2、model3。预期 model1 `21/21`、model2 `33/33`、model3 `3/3`；关闭后
均为 `0/0`。

### 3.6 batched_matmul_add 特性

- 打开：设置 `ANNC_EP_FUSION_BATCHED_MATMUL_ADD=1`（默认关闭）。
- 对应 pattern：`batched_matmul_add`；融合算子：`FusedBatchedMatmulAdd`。
- 本机 model3 含该 pattern 站点。

命令同 3.3，把开关换成 `ANNC_EP_FUSION_BATCHED_MATMUL_ADD=1/0`，跑 model3。
预期 `sites=50 claimed=50`；关闭后 `sites=0 claimed=0`。通用 `matmul` 仍可能
融合同一子图，因此“全部站点”不一定随目标 pattern 关闭而下降。

### 3.7 packed-B 特性

- 打开：设置 `ANNC_EP_PACKED_B=1`（默认关闭）；`0`、空字符串或未设置均关闭。
- model3 同时设置 `ANNC_EP_FUSION_BATCHED_MATMUL_ADD=1`；model4 同时设置
  `ANNC_EP_FUSION_TENSORDOT_MATMUL=1`。
- 在创建 ORT session 前设置；修改开关后需新建 session。

使用 3.2 的 ANNC 压测副本，分别关闭/开启 packed-B（以下为 model3，model4 则
替换模型与对应融合开关），其他配置保持相同：

```shell
for packed in 0 1; do
  ANNC_EP_FUSION_BATCHED_MATMUL_ADD=1 \
  ANNC_EP_FUSION_TENSORDOT_MATMUL=0 ANNC_EP_FUSION_LAYER_NORM=0 \
  ANNC_EP_FUSION_MASKED_MUL=0 ANNC_EP_PACKED_B="$packed" ANNC_EP_DEBUG=1 \
  numactl -C 48-63 <PYTHON> /tmp/onnx_benchmark_all_annc.py \
    --model <ONNX_MODEL_DIR>/model3.onnx --thread_num 5 --max_qps 0 \
    --max_inflight 100 --batch_size 32 --warmup_duration_s 5 \
    --test_duration_s 30 --intra_threads 16 --inter_threads 8 \
    --timeout_ms 30000 --output_md "packed-${packed}.md" --debug \
    >"packed-${packed}.log" 2>&1
done
grep -E 'packed-B state:|packed-B selected' packed-1.log
grep -E 'packed-B state:|packed-B selected' packed-0.log  # 预期无匹配
```

使能证据（需设置 `ANNC_EP_DEBUG=1`）：

- 开启时，`packed-B state:` 日志包含 `rank/slices/K/N/bytes`，确认已创建打包状态；
  推理日志出现 `bmm: packed-B selected`（model3）或
  `ftdmm: packed-B selected`（model4），确认计算实际使用 packed-B。
- 关闭时，以上两类日志均无匹配。
- 按 3.3 的方法核对 `GetCapability`：两次运行保持相同融合，model3 的
  `sites=50 claimed=50`、model4 的 `sites=15 claimed=15`，`dropped=0`。

### 3.8 使能所有特性

四个 `ANNC_EP_FUSION_*` 默认全部关闭；使能所有融合需要把四个开关都设为 `1`。
同时开启 packed-B 时另设 `ANNC_EP_PACKED_B=1`；其开关与融合独立。
基线用未修改的脚本（CPU EP），ANNC 臂用 3.2 的补丁副本：

```shell
# 基线：CPU EP
numactl -C 48-63 <PYTHON> <BENCH_SCRIPT> \
  --model <ONNX_MODEL_DIR>/model1.onnx --thread_num 5 --max_qps 0 \
  --max_inflight 100 --batch_size 32 --warmup_duration_s 5 \
  --test_duration_s 30 --intra_threads 16 --inter_threads 8 \
  --timeout_ms 30000 --output_md baseline.md

# ANNC：四个融合开关与 packed-B 全部设为 1
ANNC_EP_FUSION_TENSORDOT_MATMUL=1 ANNC_EP_FUSION_LAYER_NORM=1 \
ANNC_EP_FUSION_MASKED_MUL=1 ANNC_EP_FUSION_BATCHED_MATMUL_ADD=1 \
ANNC_EP_PACKED_B=1 \
ANNC_EP_DEBUG=1 numactl -C 48-63 <PYTHON> /tmp/onnx_benchmark_all_annc.py \
  --model <ONNX_MODEL_DIR>/model1.onnx --thread_num 5 --max_qps 0 \
  --max_inflight 100 --batch_size 32 --warmup_duration_s 5 \
  --test_duration_s 30 --intra_threads 16 --inter_threads 8 \
  --timeout_ms 30000 --output_md annc.md --debug 2>&1 | tee annc.log
```

证明：对比 `actual_qps`、`avg_latency_us`、`p99_latency_us`，并按 3.3 的方法
`grep -E 'GetCapability|site\(s\)'` 确认站点：四个开关全开时 model1/2/3/4 分别为
21/33/53/42，claimed 相同、dropped=0；默认全关时目标 pattern 为 0/0。

## 4. 常见报错

### 4.1 clang 链接 kernel 时找不到 `crtbeginS.o`

现象：

```text
/usr/bin/ld: cannot find crtbeginS.o: No such file or directory
clang: error: linker command failed with exit code 1 (use -v to see invocation)
```

根因：clang 按默认 triple `aarch64-unknown-linux-gnu` 找 GCC runtime，但目标机
GCC 实际在 `/usr/lib/gcc/aarch64-openEuler-linux/<ver>/`；`LIBRARY_PATH` 不能
替代 GCC installation 的定位。

处理：建 triple 符号链接（需要 root）。

```shell
ls -ld /usr/lib/gcc/aarch64-unknown-linux-gnu   # 先确认不存在
sudo ln -s /usr/lib/gcc/aarch64-openEuler-linux \
           /usr/lib/gcc/aarch64-unknown-linux-gnu
```

建好后重新运行；该链接让 clang 的默认 triple 找到 GCC installation 下的
`crtbeginS.o`。
