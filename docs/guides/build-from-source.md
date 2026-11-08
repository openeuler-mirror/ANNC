# 从源码构建

适用场景：需要自行编译 ANNC（本地开发、调试、定制构建选项）。依赖安装见[安装 ANNC](../getting-started/installation.md)。

## 一键构建（推荐）

```shell
./build.sh --install-prefix "$HOME/ANNC" --build-type Debug
```

LLVM/MLIR 使用外部工具链（CMake `find_package`，不锁定版本），不再下载或编译 LLVM；nlohmann/json 由 CMake 在 configure 时 `FetchContent` 获取到 `third_party/json/`。openEuler 24.03 用系统 `llvm-toolset-20`；openEuler 22.03 仓库只有 LLVM/MLIR 12.0.1，需自建 LLVM/MLIR 20/21 并通过 `ANNC_LLVM_BIN` 或 `-DLLVM_DIR`/`-DMLIR_DIR` 指定，见[安装 ANNC](../getting-started/installation.md#4-llvm--mlir-与-nlohmannjson)。

RELEASE 模式下 KDNN 自动从 release zip 下载并解压（默认构建不启用 KDNN 后端，需同时打开 adaptor）：

```shell
./build.sh --enable-kdnn-adaptor --kdnn-source RELEASE --kdnn-lib-variant sve-threadpool
```

`--kdnn-source` 只决定 KDNN 源码/预编译包的来源；未设置 `--enable-kdnn-adaptor`（即 CMake `ANNC_ENABLE_KDNN_ADAPTOR=OFF`，默认）时 KDNN 不会进入构建，得到的产物不含 KDNN 后端。

只调试 ANNC 代码、不重编 third_party（json）：

```shell
./build.sh --annc-debug
```

## 手动构建

手动构建不会经过 `build.sh` 的 proto 预生成步骤；在干净检出上先运行（关闭对应前端时可跳过相应脚本）：

```shell
bash tf_protos_minimal.sh -d "$PWD"
bash frontends/onnx/protos/onnx_protos_minimal.sh -d "$PWD"
```

LLVM/MLIR 的查找顺序为 `-DLLVM_DIR`/`-DMLIR_DIR` → 环境变量 `ANNC_LLVM_BIN` → `LLVM_ROOT`/`MLIR_ROOT` → `llvm-config-{22,21,20} --cmakedir`。标准 `llvm-toolset-20` 安装且 `llvm-config-20` 在 `PATH` 时无需导出；工具链在自定义前缀时：

```shell
export ANNC_LLVM_BIN=<LLVM 20 的 bin 目录>
# 或：export LLVM_DIR=<.../lib64/cmake/llvm>
#     export MLIR_DIR=<.../lib64/cmake/mlir>
```

自建的安装树同样适用。若直接使用 LLVM 构建树而不安装，MLIR 的配置位于 `<llvm-build>/tools/mlir/cmake/modules`，需显式传 `-DMLIR_DIR`（该布局不在 `ANNC_LLVM_BIN` 的前缀推断范围内）。

然后配置并构建。默认启用 TensorFlow addon，ABI 和 C++ 标准自动检测：

```shell
mkdir -p build && cd build
cmake -G Ninja .. \
  -DCMAKE_INSTALL_PREFIX="$HOME/ANNC" \
  -DCMAKE_BUILD_TYPE=Debug
ninja && ninja install
```

## `build.sh` 常用选项

- `--build-type [Debug|Release|RelWithDebInfo]`：构建类型（默认 Release）
- `--install-prefix <path>`：安装路径（默认 `./install`）
- `--gemm-package-dir <path>`：指定解压后的 GEMM-ASM 完整包根目录（默认 `third_party/annc-gemm-kernels`；对应 CMake 选项 `ANNC_GEMM_PACKAGE_DIR`）
- `--require-gemm-package`：CMake 配置时强制要求完整 GEMM-ASM 包可用；缺包或版本不兼容时配置失败（对应 `ANNC_REQUIRE_GEMM_PACKAGE=ON`）
- `--enable-python` / `--disable-python`：构建 / 不构建 ANNC Python 绑定（默认关闭；开启时自动安装 pybind11 / nanobind）
- `--clean`：清理构建目录后重新构建
- `--no-install-deps`：跳过 pybind11 / nanobind 自动安装（仅 `--enable-python` 相关）
- `--annc-debug`：只以调试模式编译 ANNC 代码（`-g3 -O0 -UNDEBUG`），third_party 保持原构建类型、不重编
- 框架选择（互斥）：`--enable-tf` 只构建 TF 前端/addon；`--enable-ort`（兼容别名 `--enable-ort-ep`）只构建 ORT 加速栈——ONNX 前端与 ORT EP 视为一个整体，总是同时构建；`--all` 构建 TF + ORT 加速栈。不传时保持历史默认（TF + ONNX 前端，ORT EP 关闭）
- `--dry-run`：打印解析后的框架开关并退出，不 configure/build
- `--disable-tensorflow`：关闭 TensorFlow 前端（`annc-tf2atir`、`annc-converter`、`annc-tf-pipeline`）及其 addon（legacy/custom 模式；与 `--enable-tf`/`--all` 冲突）
- `--disable-tensorflow-addon`：只关闭 TF addon（`annc_optimizer`、`annc_fused_op`），TF 前端工具仍构建（legacy/custom 模式；与 `--enable-tf`/`--all` 冲突）
- `--disable-onnx`：关闭 ONNX 前端（`annc-onnx2atir`）（legacy/custom 模式；与 `--enable-ort`/`--all` 冲突）
- `--kdnn-source [LOCAL|REMOTE|RELEASE]`：KDNN 来源（`build.sh` 默认 `RELEASE`，CMake 默认 `LOCAL`）；`LOCAL` 用本地 `third_party/KDNN`，`REMOTE` 从 git 仓库 FetchContent
- `--enable-kdnn-adaptor` / `--disable-kdnn-adaptor`：启用/关闭内置 KDNN adaptor kernel（默认关闭）；关闭时 KDNN 来源选项不产生实际集成
- `--kdnn-lib-variant <variant>`：RELEASE 模式下选择库变体（默认 `sve-threadpool`）
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
| `ANNC_ENABLE_PYTHON` | `ON`/`OFF` | `OFF` | 构建 ANNC Python 绑定；开启时需要 pybind11/nanobind 与 `llvm-toolset-20-python3-mlir` 提供的 MLIR Python 绑定源码；对应 `build.sh --enable-python` |
| `ANNC_ONNXRUNTIME_INCLUDE_DIR` | 路径 | 空 | ORT 1.26.0 的 session 头文件目录（含 `onnxruntime_c_api.h`、`onnxruntime_ep_c_api.h`）；EP 开启时必填，缺失即 configure 失败，不自动下载 SDK |
| `ANNC_EP_ORT_BUILD_TREE` | 路径 | 空 | 宿主 ORT 构建树的 `Release/` 目录，提供原生池所需的 common/cpuinfo 及 packed-B 所需的 MLAS/KleidiAI 四个归档；EP 开启时必填 |
| `ANNC_KERNEL_MLAS` | `ON`/`OFF` | `OFF`（`ANNC_ENABLE_ONNXRUNTIME_EP=ON` 时强制 `ON`） | 启用 MLAS GEMM kernel 后端；构建 ORT EP 时自动启用，无需手动设置；构建期需 `ANNC_MLAS_INCLUDE_DIR` 与宿主 MLAS/KleidiAI 归档，运行期由 `ANNC_EP_ORT_BUILD_TREE` 解析四个归档 |
| `ANNC_MLAS_INCLUDE_DIR` | 路径 | 空 | ORT 源码树的 `onnxruntime/core/mlas/inc`（含 `mlas.h`）；`ANNC_KERNEL_MLAS=ON` 时必填 |
| `ANNC_GEMM_PACKAGE_DIR` | 路径 | `third_party/annc-gemm-kernels` | GEMM-ASM 完整二进制包目录；包含基础归档、codegen 共享库、公开头文件、CMake 配置和 manifest，配置时校验 API 1、kernel ABI 1 及包内文件哈希；无需私有源码 |
| `ANNC_REQUIRE_GEMM_PACKAGE` | `ON`/`OFF` | `OFF` | 可选的构建期强校验；适合 CI、发布构建等场景。开启时 CMake configure 阶段缺少兼容完整 GEMM-ASM 包即失败，避免运行时才报错；关闭时允许只构建普通工具，但 GEMM microkernel、prepack 和 epilogue 请求会明确报错。`build.sh` 对应 `--require-gemm-package` |
| `ANNC_ENABLE_KDNN_ADAPTOR` | `ON`/`OFF` | `OFF` | 构建 KDNN 后端的内置 kernel adaptor；对应 `build.sh --enable-kdnn-adaptor`，仅 AArch64 |
| `ANNC_KDNN_SOURCE` | `LOCAL`/`REMOTE`/`RELEASE` | `LOCAL`（`build.sh` 默认覆盖为 `RELEASE`） | KDNN 来源模式；仅在 `ANNC_ENABLE_KDNN_ADAPTOR=ON` 时参与集成 |
| `ANNC_KDNN_DIR` | 路径 | `third_party/KDNN` | LOCAL 模式下 KDNN 树根目录 |
| `ANNC_KDNN_RELEASE_DIR` | 路径 | `third_party/kdnn-release` | RELEASE 模式下 release 包解压与整理目录 |
| `ANNC_KDNN_RELEASE_URL` | URL | `https://gitcode.com/boostkit/boostsra/releases/download/v1.2.0/BoostKit-boostcore-kdnn_3.1.0.zip` | RELEASE 模式 zip 包下载地址 |
| `ANNC_KDNN_RELEASE_SHA256` | 字符串 | `61a4b0b55a80ca742b43dde638b7fdd63c7ef36f26f3615e4f1ad008750217a8` | RELEASE 模式 zip 包 SHA256 校验值；留空则跳过校验 |
| `ANNC_KDNN_LIB_VARIANT` | 字符串 | `sve-threadpool` | RELEASE 模式下库变体，可选 `sve-threadpool`、`sve-omp`、`sve2-threadpool`、`sve2-omp` |
| `ANNC_ENABLE_TF2ATIR_V2` | `ON`/`OFF` | `OFF` | 构建 TF stablehlo V2 路线：stablehlo 方言集成与 sink 层（v1.9.7 Embedded，要求 LLVM/MLIR 20.1.8 工具链与 `third_party/stablehlo`）与 V2 工具（`annc-converter-v2` 随批次合入；`annc-tf2atir-v2` 同开关），见下文「TF stablehlo V2 路线」 |

所有框架组合都会构建通用 `annc-partition` 查询入口。当前该目标仅提供占位流程（各阶段返回 `not implemented`），不注册真实 TF 切分能力，也不改变既有前端、addon 或后端开关。

### GEMM-ASM 完整包

`ANNC_GEMM_PACKAGE_DIR` 指定完整 GEMM-ASM 包，默认路径为
`third_party/annc-gemm-kernels`。包内基础归档、codegen 共享库、公开头文件、
CMake 配置和 manifest 必须作为一个版本化整体存在；不再支持单独指定或替换
基础 `.a`。缺包时普通编译工具仍可构建，但 GEMM microkernel、静态 RHS
预打包和 epilogue 请求会明确报错。补齐完整包后须重新配置和构建。

包解压在其他目录时，可通过 `build.sh` 或直接调用 CMake 指定 package 根目录。
使用 `build.sh` 时：

```shell
./build.sh --gemm-package-dir /opt/annc-gemm-kernels
```

该路径会传入 CMake 的 `ANNC_GEMM_PACKAGE_DIR`。复用已有 `build/` 时若更改包路径，
需加 `--clean` 重新配置：

```shell
./build.sh --clean --gemm-package-dir /opt/annc-gemm-kernels
```

直接使用 CMake 时：

```shell
cmake -S . -B build -G Ninja \
  -DANNC_GEMM_PACKAGE_DIR=/opt/annc-gemm-kernels \
  -DANNC_TENSORFLOW_CXX11_ABI=1
cmake --build build
cmake --install build
```

CI 或发布构建需要在配置阶段拒绝缺包时，`build.sh` 可组合使用两个选项：

```shell
./build.sh --gemm-package-dir /opt/annc-gemm-kernels --require-gemm-package
```

直接使用 CMake 时等价于添加 `-DANNC_REQUIRE_GEMM_PACKAGE=ON`。
复用已有 `build/` 时，若更改强制检查开关，也需加 `--clean` 重新配置。

路径必须直接包含 `manifest.json`、`cmake/`、`include/` 和对应库文件，不能指向
`annc-gemm-kernels.tar.gz`。安装时整包复制到
`<prefix>/<libdir>/annc-gemm-kernels/`；安装后的 ANNC 编译工具通过相对 RPATH
加载 codegen 库，无需设置运行时包路径环境变量。

### TF stablehlo V2 路线（sink 层与 V2 工具）

`ANNC_ENABLE_TF2ATIR_V2=ON` 开启整条 V2 路线（stablehlo 集成与 sink 层，
V2 工具并入同一开关）。stablehlo 集成为 v1.9.7 的 Embedded 模式：
stablehlo 以 `EXCLUDE_FROM_ALL` 子目录加入构建，并通过全局属性快照/还原
使只有被 ANNC 目标显式链接的 stablehlo 库参与编译；默认构建（OFF）完全
不触碰 stablehlo。该层是 TF stablehlo 前端（pb2stablehlo）与 stablehlo
sink 转换层的方言依赖，要求外部工具链为 LLVM/MLIR 20.1.8（stablehlo
v1.9.x 的配套版本）。

依赖获取（不入库，`third_party/*` 已被忽略）：

```shell
git -C third_party clone --depth 1 --branch v1.9.7 \
  https://github.com/openxla/stablehlo.git stablehlo
```

构建与验证（推荐使用独立构建目录，与默认 `build/` 互不干扰；
`build-stablehlo/` 已被忽略）：

```shell
export ANNC_LLVM_BIN=/home/crf/dev/annc/llvm20/bin
cmake -S . -B build-stablehlo -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$(command -v gcc) \
  -DCMAKE_CXX_COMPILER=$(command -v g++) \
  -DCMAKE_CXX_FLAGS=-fPIC \
  -DANNC_ENABLE_TF2ATIR_V2=ON \
  -DLLVM_DIR=/home/crf/dev/annc/llvm20/lib/cmake/llvm \
  -DMLIR_DIR=/home/crf/dev/annc/llvm20/lib/cmake/mlir
cmake --build build-stablehlo --target \
  "stablehlo/stablehlo/dialect/libStablehloOps.a"
```

configure 成功时输出 `ANNC: stablehlo artifact sink layer enabled
(v1.9.x embedded)`；ANNC 当前仅消费 stablehlo 方言/变换核心库族
（如 `libStablehloOps.a`）。stablehlo 自带的 `stablehlo-opt` 等工具会
连带编译 `conversions/tosa` 等转换，其 TOSA 代码面向 stablehlo v1.9.x
配套的 LLVM 21 API，在 20.1.8 工具链下不保证可编译，且非 ANNC 消费
目标，不作为验证口径。

## 验证安装

```shell
export PATH="$HOME/ANNC/bin:$PATH"

which annc-opt annc-asm                 # 工具是否可用
python3 -m pytest tests/                # Python 测试
cd build && ctest --output-on-failure   # C++ 单元测试（GTest，随开关注册）
```

### 覆盖率报告（gcov）

覆盖率插桩只作用于 ANNC 作用域（LLVM 为外部工具链，不插桩）。但 `build.sh` 的缓存一致性检查会把 `ANNC_ENABLE_COVERAGE` 与既有 `CMakeCache.txt` 比对：在已存在的构建目录上直接加 `--coverage` 会被拒绝并要求 `--clean`，而 `--clean` 会清空整个 `build/`（含 third_party/json，将重新获取）。

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
- LOCAL / RELEASE 自动检测 KDNN packed-B 接口及链接符号，仅在可用时构建对应适配器；ANNC 原生预打包能力始终构建，不受影响。
- REMOTE 源码构建自动应用 `patches/kdnn_rhs_packed` 补丁，提供 packed-B 接口。
- `--build-type Debug` 会把 third_party（json）一起重编；只调试 ANNC 时改用 `--annc-debug`。
- 切换 Python 环境或 TensorFlow 版本后需重新运行 CMake（configure 时自动检测 TF）。
