# 安装 ANNC

目标：在 openEuler（AArch64）上装齐 ANNC 的构建依赖，并确认工具链可用。构建与安装命令见[从源码构建](../guides/build-from-source.md)。

## 环境要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | openEuler-24.03-LTS-SP4 |
| 处理器 | 鲲鹏 920X（AArch64） |
| LLVM / MLIR | 系统 `llvm-toolset-20`（本机验证 20.1.8，不锁定版本）；CMake `find_package(LLVM CONFIG)` / `find_package(MLIR CONFIG)` 使用预装工具链 |
| 磁盘空间 | ANNC 构建目录约数 GB；不再下载或编译 LLVM |

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

```shell
sudo yum install -y cmake ninja-build gcc gcc-c++ python3-devel openssl-devel \
  gtest-devel unzip rpm cpio
sudo yum install -y protobuf-devel protobuf-compiler
sudo yum install -y llvm-toolset-20-llvm llvm-toolset-20-llvm-devel \
  llvm-toolset-20-llvm-libs llvm-toolset-20-llvm-static \
  llvm-toolset-20-llvm-mlir llvm-toolset-20-llvm-mlir-devel \
  llvm-toolset-20-llvm-mlir-static llvm-toolset-20-python3-mlir \
  llvm-toolset-20-clang
```

| 依赖 | 用途 |
| --- | --- |
| `cmake` / `ninja-build` | 构建系统（`build.sh` 默认使用 Ninja 生成器） |
| `gcc` / `gcc-c++` | 默认 C/C++ 编译器（`build.sh` 的 `CC`/`CXX` 默认值） |
| `llvm-toolset-20-*` | LLVM/MLIR 20 工具链：CMake 配置、MLIR 头文件/库、静态归档、`clang`、Python 绑定源码 |
| `python3-devel` | pybind11 / nanobind 扩展需要的 `Python.h`（仅 `--enable-python`） |
| `gtest-devel` | C++ 单元测试（CTest） |
| `unzip` / `rpm`（提供 `rpm2cpio`） / `cpio` | KDNN RELEASE 模式解包 release zip；`build.sh` 默认 `--kdnn-source RELEASE`，在 CMake 之前检查这三个命令 |
| `openssl-devel` | JIT cache key 哈希（OpenSSL Crypto）；仅 `ANNC_ENABLE_TENSORFLOW_ADDON=ON` 时需要 |
| `protobuf-devel` / `protobuf-compiler` | 启用 TF 或 ONNX 前端时必需；`protoc` 用于校验与重新生成 proto 源码。ONNX 侧缺失即报错；TF 侧会先尝试自动下载 fallback protoc |

### 4. LLVM / MLIR 与 nlohmann/json

LLVM/MLIR 由系统 `llvm-toolset-20-*` 提供，CMake 用 `find_package(LLVM CONFIG)` / `find_package(MLIR CONFIG)` 使用预装工具链，不再下载或编译 LLVM。标准安装路径（或 `llvm-config-20` 在 `PATH`）时无需额外环境变量；工具链在自定义前缀时导出 `ANNC_LLVM_BIN=<LLVM 20 bin 目录>`，或传 `-DLLVM_DIR`/`-DMLIR_DIR`。

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
