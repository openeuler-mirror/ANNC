# 安装 ANNC

目标：在 openEuler（AArch64）上装齐 ANNC 的构建依赖，并确认工具链可用。构建与安装命令见[从源码构建](../guides/build-from-source.md)。

## 环境要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | openEuler-24.03-LTS-SP4（系统 `llvm-toolset-20`）；openEuler-22.03-LTS-SP4（需自建 LLVM/MLIR 20/21） |
| 处理器 | 鲲鹏 920X（AArch64） |
| LLVM / MLIR | LLVM/MLIR 20 或 21（24.03 基线 20.1.8，22.03 自建示例 21.1.3，不锁定小版本）；CMake `find_package(LLVM CONFIG)` / `find_package(MLIR CONFIG)` 使用外部工具链 |
| 磁盘空间 | ANNC 构建目录约数 GB；22.03 自建 LLVM/MLIR 另需更多空间 |

## 前置条件

- 可访问 Gitee 镜像：CMake 通过 `FetchContent` 获取 nlohmann/json（header-only）。
- 具备 `sudo` 权限安装系统包。
- Python 3 及 `pip3`。

## 步骤

### 1. 获取源码

```shell
git clone https://gitcode.com/openeuler/ANNC.git --branch dev
cd ANNC
```

下文命令均在本仓库根目录下执行。

### 2. 安装 Python 依赖

```shell
pip3 install -r requirements.txt
```

包含 `pybind11`、`nanobind`、`numpy`、`pytest` 与 `tensorflow`：`tensorflow` 仅在构建 TF addon 时需要；`pybind11`/`nanobind` 仅在 `--enable-python` 构建 Python 绑定时需要（默认不构建）。

### 3. 安装系统依赖

基础系统依赖（两种 OS 相同）：

```shell
sudo yum install -y cmake ninja-build gcc gcc-c++ python3-devel \
  gtest-devel unzip rpm cpio
sudo yum install -y protobuf-devel protobuf-compiler
```

LLVM/MLIR 按 OS 版本二选一。openEuler 24.03 直接安装系统工具链：

```shell
sudo yum install -y llvm-toolset-20-llvm llvm-toolset-20-llvm-devel \
  llvm-toolset-20-llvm-libs llvm-toolset-20-llvm-static \
  llvm-toolset-20-llvm-mlir llvm-toolset-20-llvm-mlir-devel \
  llvm-toolset-20-llvm-mlir-static llvm-toolset-20-python3-mlir \
  llvm-toolset-20-clang
```

openEuler 22.03 没有 `llvm-toolset-20`：everything 仓库只提供 LLVM/MLIR 12.0.1，EPOL 只提供 `llvm15`/`clang15`（不含 MLIR），均不满足 ANNC 所需的 MLIR 20/21，需按步骤 4 自建。

| 依赖 | 用途 |
| --- | --- |
| `cmake` / `ninja-build` | 构建系统（`build.sh` 默认使用 Ninja 生成器） |
| `gcc` / `gcc-c++` | 默认 C/C++ 编译器（`build.sh` 的 `CC`/`CXX` 默认值） |
| `llvm-toolset-20-*` | 仅 openEuler 24.03：LLVM/MLIR 20 工具链，含 CMake 配置、MLIR 头文件/库、静态归档、`clang`、Python 绑定源码 |
| 自建 LLVM/MLIR | 仅 openEuler 22.03：仓库无可用包，见步骤 4 |
| `python3-devel` | pybind11 / nanobind 扩展需要的 `Python.h`（仅 `--enable-python`） |
| `gtest-devel` | C++ 单元测试（CTest） |
| `unzip` / `rpm`（提供 `rpm2cpio`） / `cpio` | KDNN RELEASE 模式解包 release zip；`build.sh` 默认 `--kdnn-source RELEASE`，在 CMake 之前检查这三个命令 |
| `protobuf-devel` / `protobuf-compiler` | 启用 TF 或 ONNX 前端时必需；`protoc` 用于校验与重新生成 proto 源码。ONNX 侧缺失即报错；TF 侧会先尝试自动下载 fallback protoc |

### 4. LLVM / MLIR 与 nlohmann/json

ANNC 需要 LLVM/MLIR 20 或 21（不锁定小版本；`annc/include/Support/LLVMCompat.h` 处理 21 的 API 差异），由 CMake `find_package(LLVM CONFIG)` / `find_package(MLIR CONFIG)` 使用外部工具链。

**openEuler 24.03**：使用系统 `llvm-toolset-20-*`。标准安装路径（或 `llvm-config-20` 在 `PATH`）时无需额外环境变量；自定义前缀时导出 `ANNC_LLVM_BIN=<LLVM bin 目录>`，或传 `-DLLVM_DIR`/`-DMLIR_DIR`。

**openEuler 22.03**：仓库的 LLVM/MLIR 12.0.1 与 ANNC 所需的 MLIR 20/21 跨越多个大版本，不能直接使用，需自建（本机以 llvm-project 21.1.3 验证）。在 llvm-project monorepo 根目录执行（`-S` 指向 `llvm/` 子目录）：

```shell
cmake -G Ninja -S llvm -B llvm-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/llvm" \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_TARGETS_TO_BUILD="AArch64" \
  -DLLVM_ENABLE_ASSERTIONS=OFF
ninja -C llvm-build install
```

安装后让 CMake 找到它，两种方式任选：

```shell
export ANNC_LLVM_BIN="$HOME/llvm/bin"   # build.sh 与 CMake 均可识别

# 或手动 configure 时显式指定（安装目录为 lib64 时把 lib 换成 lib64）
cmake ... \
  -DLLVM_DIR="$HOME/llvm/lib/cmake/llvm" \
  -DMLIR_DIR="$HOME/llvm/lib/cmake/mlir"
```

若直接复用 LLVM 构建树而不安装，MLIR 的配置目录是 `<llvm-build>/tools/mlir/cmake/modules`，需显式传 `-DMLIR_DIR`。`--enable-python` 还要求自建产物包含共享 MLIR 库与 MLIR Python 绑定（`-DMLIR_ENABLE_BINDINGS_PYTHON=ON`），否则保持 Python 绑定关闭。

nlohmann/json 仍由 CMake `FetchContent` 在 configure 时获取到 `third_party/json/`（header-only）。

运行期 `annc`/`annc-onnx-pipeline` 从 `ANNC_LLVM_BIN`（缺省为构建期记录值）解析 `mlir-opt`/`mlir-translate`/`opt`/`llc`/`clang`；目标机工具链路径与构建机不同时，需在运行环境导出。

> `build.sh --enable-python` 会在缺失时自动安装 `pybind11` / `nanobind`（默认不构建 Python 绑定）；传入 `--no-install-deps` 可关闭自动安装。TensorFlow 需自行安装。

## 验证

```shell
./build.sh --install-prefix "$HOME/ANNC" --build-type Debug   # 详细选项见「从源码构建」
export PATH="$HOME/ANNC/bin:$PATH"
which annc-opt annc-asm
```

能解析出可执行文件路径即安装成功。构建选项、测试与覆盖率见[从源码构建](../guides/build-from-source.md)。
