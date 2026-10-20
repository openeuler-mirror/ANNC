# 从源码构建

适用场景：需要自行编译 ANNC（本地开发、调试、定制构建选项）。依赖安装见[安装 ANNC](../getting-started/installation.md)。

## 一键构建（推荐）

```shell
./build.sh --install-prefix "$HOME/ANNC" --build-type Debug
```

LLVM/MLIR 与 nlohmann/json 由 CMake 在 configure 时自动拉取到 `third_party/` 并从源码编译，无需手动准备。

RELEASE 模式下 KDNN 自动从 release zip 下载并解压（默认构建不启用 KDNN 后端，需同时打开 adaptor）：

```shell
./build.sh --enable-kdnn-adaptor --kdnn-source RELEASE --kdnn-lib-variant sve-threadpool
```

`--kdnn-source` 只决定 KDNN 源码/预编译包的来源；未设置 `--enable-kdnn-adaptor`（即 CMake `ANNC_ENABLE_KDNN_ADAPTOR=OFF`，默认）时 KDNN 不会进入构建，得到的产物不含 KDNN 后端。

只调试 ANNC 代码、不重编 LLVM：

```shell
./build.sh --annc-debug
```

## 手动构建

手动构建不会经过 `build.sh` 的 proto 预生成步骤；在干净检出上先运行（关闭对应前端时可跳过相应脚本）：

```shell
bash tf_protos_minimal.sh -d "$PWD"
bash frontends/onnx/protos/onnx_protos_minimal.sh -d "$PWD"
```

然后配置并构建。默认启用 TensorFlow addon，必须显式指定 `ANNC_TENSORFLOW_CXX11_ABI`：

```shell
mkdir -p build && cd build
cmake -G Ninja .. \
  -DCMAKE_INSTALL_PREFIX="$HOME/ANNC" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DANNC_TENSORFLOW_CXX11_ABI=1
ninja && ninja install
```

## `build.sh` 常用选项

- `--build-type [Debug|Release|RelWithDebInfo]`：构建类型（默认 Release）
- `--install-prefix <path>`：安装路径（默认 `./install`）
- `--enable-libcxx`：启用 libc++
- `--clean`：清理构建目录后重新构建
- `--no-install-deps`：跳过 pybind11 / nanobind 自动安装
- `--annc-debug`：只以调试模式编译 ANNC 代码（`-g3 -O0 -UNDEBUG`），third_party 保持原构建类型、不重编
- 框架选择（互斥）：`--enable-tf` 只构建 TF 前端/addon；`--enable-ort`（兼容别名 `--enable-ort-ep`）只构建 ORT 加速栈——ONNX 前端与 ORT EP 视为一个整体，总是同时构建；`--all` 构建 TF + ORT 加速栈。不传时保持历史默认（TF + ONNX 前端，ORT EP 关闭）
- `--dry-run`：打印解析后的框架开关并退出，不 configure/build
- `--disable-tensorflow`：关闭 TensorFlow 前端（`annc-tf2atir`、`annc-converter`、`annc-tf-pipeline`）及其 addon（legacy/custom 模式；与 `--enable-tf`/`--all` 冲突）
- `--disable-tensorflow-addon`：只关闭 TF addon（`annc_optimizer`、`annc_fused_op`），TF 前端工具仍构建（legacy/custom 模式；与 `--enable-tf`/`--all` 冲突）
- `--disable-onnx`：关闭 ONNX 前端（`annc-onnx2atir`）（legacy/custom 模式；与 `--enable-ort`/`--all` 冲突）
- `--kdnn-source [LOCAL|REMOTE|RELEASE]`：KDNN 来源（`build.sh` 默认 `RELEASE`，CMake 默认 `LOCAL`）；`LOCAL` 用本地 `third_party/KDNN`，`REMOTE` 从 git 仓库 FetchContent
- `--enable-kdnn-adaptor` / `--disable-kdnn-adaptor`：启用/关闭内置 KDNN adaptor kernel（默认关闭）；关闭时 KDNN 来源选项不产生实际集成
- `--kdnn-lib-variant <variant>`：RELEASE 模式下选择库变体（默认 `sve-threadpool`）
- `--enable-constant-folding`：启用常量折叠与 KDNN packed-B 支持（默认关闭）
- `--disable-assertions`：关闭断言（默认开启断言）
- `--kdnn-dir <path>`：LOCAL 模式下的本地 KDNN 树（默认 `third_party/KDNN`；仅 `--kdnn-source LOCAL` 有效）
- `--regen-tf-protos` / `--regen-onnx-protos`：强制重新生成 TF / ONNX protobuf 源码
- `--coverage`：以覆盖率插桩方式配置；首次启用需配合 `--clean`（会重建 `build/`），复用既有构建目录的绕过方式见[覆盖率报告](#覆盖率报告gcov)

## CMake 选项参考

| CMake 选项 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `ANNC_DEBUG` | `ON`/`OFF` | `OFF` | 只以调试模式编译 ANNC 自有代码（`-g3 -O0 -UNDEBUG`），third_party 构建类型不变、不重编；对应 `build.sh --annc-debug` |
| `ANNC_ENABLE_TENSORFLOW` | `ON`/`OFF` | `ON` | 构建 TF 前端工具（`annc-tf2atir`、`annc-converter`、`annc-tf-pipeline`）并编译预生成的 TF proto（proto 由 `build.sh` 或脚本预生成）；对应 `build.sh --disable-tensorflow` |
| `ANNC_ENABLE_TENSORFLOW_ADDON` | `ON`/`OFF` | 跟随 `ANNC_ENABLE_TENSORFLOW` | 构建并安装 TF addon（`annc_optimizer`、`annc_fused_op`、`annc_jit_cache`）及其测试；为 `ON` 要求 `ANNC_ENABLE_TENSORFLOW=ON`；对应 `build.sh --disable-tensorflow-addon` |
| `ANNC_ENABLE_ONNX` | `ON`/`OFF` | `ON` | 构建 ONNX 前端（`annc-onnx2atir` 及其 GTest）并编译预生成的 ONNX proto（proto 由 `build.sh` 或脚本预生成）；对应 `build.sh --disable-onnx` |
| `ANNC_ENABLE_ONNXRUNTIME_EP` | `ON`/`OFF` | `OFF` | 构建并安装 ONNX Runtime EP 插件 `libannc_ep.so`（`addons/onnxruntime/`）；要求 `ANNC_ENABLE_ONNX=ON`，对应 `build.sh --enable-ort`/`--all` |
| `ANNC_ONNXRUNTIME_INCLUDE_DIR` | 路径 | 空 | ORT 1.26.0 的 session 头文件目录（含 `onnxruntime_c_api.h`、`onnxruntime_ep_c_api.h`）；EP 开启时必填，缺失即 configure 失败，不自动下载 SDK |
| `ANNC_EP_ORT_BUILD_TREE` | 路径 | 空 | 宿主 ORT 构建树的 `Release/` 目录，提供原生池所需的 `libonnxruntime_common.a` 与 `libcpuinfo.a`；EP 开启时必填 |
| `ANNC_KERNEL_MLAS` | `ON`/`OFF` | `OFF`（`ANNC_ENABLE_ONNXRUNTIME_EP=ON` 时强制 `ON`） | 启用 MLAS GEMM kernel 后端；构建 ORT EP 时自动启用，无需手动设置；构建期需 `ANNC_MLAS_INCLUDE_DIR`，运行期由 `ANNC_MLAS_LIB_DIR` 或 `ANNC_KERNEL_LIB_PATH` 解析四个归档 |
| `ANNC_MLAS_INCLUDE_DIR` | 路径 | 空 | ORT 源码树的 `onnxruntime/core/mlas/inc`（含 `mlas.h`）；`ANNC_KERNEL_MLAS=ON` 时必填 |
| `ANNC_GEMM_ASM_SOURCE_DIR` | 路径 | 空 | GEMM-ASM checkout（含 `Sve/`、`Neon/`）；作为 `annc` driver 汇编生成微内核的编译期默认值，运行时同名环境变量可覆盖；生成 `model_generated_kernels/*.S` 时未配置则 `annc` 报错退出 |
| `ANNC_ENABLE_KDNN_ADAPTOR` | `ON`/`OFF` | `OFF` | 构建 KDNN 后端的内置 kernel adaptor；对应 `build.sh --enable-kdnn-adaptor`，仅 AArch64 |
| `ANNC_KDNN_SOURCE` | `LOCAL`/`REMOTE`/`RELEASE` | `LOCAL`（`build.sh` 默认覆盖为 `RELEASE`） | KDNN 来源模式；仅在 `ANNC_ENABLE_KDNN_ADAPTOR=ON` 时参与集成 |
| `ANNC_KDNN_DIR` | 路径 | `third_party/KDNN` | LOCAL 模式下 KDNN 树根目录 |
| `ANNC_KDNN_RELEASE_DIR` | 路径 | `third_party/kdnn-release` | RELEASE 模式下 release 包解压与整理目录 |
| `ANNC_KDNN_RELEASE_URL` | URL | `https://gitcode.com/boostkit/boostsra/releases/download/v1.2.0/BoostKit-boostcore-kdnn_3.1.0.zip` | RELEASE 模式 zip 包下载地址 |
| `ANNC_KDNN_RELEASE_SHA256` | 字符串 | `61a4b0b55a80ca742b43dde638b7fdd63c7ef36f26f3615e4f1ad008750217a8` | RELEASE 模式 zip 包 SHA256 校验值；留空则跳过校验 |
| `ANNC_KDNN_LIB_VARIANT` | 字符串 | `sve-threadpool` | RELEASE 模式下库变体，可选 `sve-threadpool`、`sve-omp`、`sve2-threadpool`、`sve2-omp` |

## 验证安装

```shell
export PATH="$HOME/ANNC/bin:$PATH"

which annc-opt annc-asm                 # 工具是否可用
python3 -m pytest tests/                # Python 测试
cd build && ctest --output-on-failure   # C++ 单元测试（GTest，随开关注册）
```

### 覆盖率报告（gcov）

覆盖率插桩只作用于 ANNC 作用域（在 LLVM 子构建之后启用，LLVM 不插桩）。但 `build.sh` 的缓存一致性检查会把 `ANNC_ENABLE_COVERAGE` 与既有 `CMakeCache.txt` 比对：在已存在的构建目录上直接加 `--coverage` 会被拒绝并要求 `--clean`，而 `--clean` 会清空整个 `build/`（含 third_party，LLVM 将重编）。

```shell
./build.sh --coverage --clean      # 首次启用覆盖率（会重建 build/）
cd build && ninja coverage
# HTML 报告：build/coverage/index.html
```

若要复用既有 `build/`、只重编 ANNC，可跳过 `build.sh`，直接在该目录重配：

```shell
cd build && cmake -DANNC_ENABLE_COVERAGE=ON .. && ninja
```

Release 下行覆盖准确、分支覆盖为近似值；需精确分支覆盖须清理后用 Debug 配置：`rm -rf build && ./build.sh --coverage --build-type Debug`。

## 注意事项

- KDNN backend（`ANNC_ENABLE_KDNN_ADAPTOR=ON`）仅支持 AArch64 平台。
- RELEASE 模式需要系统安装 `unzip`、`rpm2cpio`、`cpio`（openEuler：`sudo yum install -y unzip rpm cpio`）。
- RELEASE 且 `ANNC_ENABLE_KDNN_ADAPTOR=ON` 时当前不支持 `ANNC_ENABLE_CONSTANT_FOLDING=ON`。
- `--build-type Debug` 会把 third_party（LLVM/json）一起重编；只调试 ANNC 时改用 `--annc-debug`。
- 切换 Python 环境或 TensorFlow 版本后需重新运行 CMake（configure 时自动检测 TF）。
