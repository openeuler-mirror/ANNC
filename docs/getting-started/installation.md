# 安装 ANNC

目标：在 openEuler（AArch64）上装齐 ANNC 的构建依赖，并确认工具链可用。构建与安装命令见[从源码构建](../guides/build-from-source.md)。

## 环境要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | openEuler-24.03-LTS-SP4 |
| 处理器 | 鲲鹏 920X（AArch64） |
| LLVM / MLIR | 21.1.3（CMake `FetchContent` 自动下载并从源码编译，无需手动准备） |
| 磁盘空间 | 约 50GB；首次编译 LLVM 需 30–60 分钟，增量构建仅重编变更文件 |

## 前置条件

- 可访问 Gitee 镜像：CMake 从镜像下载 LLVM/MLIR 21.1.3 与 nlohmann/json 源码。
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

包含 `pybind11`、`nanobind`、`numpy`、`pytest` 与 `tensorflow`（TensorFlow 仅在构建 TF addon 时需要）。

### 3. 安装系统依赖

```shell
sudo yum install -y cmake clang ninja-build gcc gcc-c++ python3-devel openssl-devel gtest-devel unzip rpm cpio
sudo yum install -y protobuf-devel protobuf-compiler
```

| 依赖 | 用途 |
| --- | --- |
| `cmake` / `ninja-build` | 构建系统（`build.sh` 默认使用 Ninja 生成器） |
| `gcc` / `gcc-c++` | 默认 C/C++ 编译器（`build.sh` 的 `CC`/`CXX` 默认值） |
| `clang` | `annc` driver 链接阶段调用，可用 `ANNC_CLANG` 覆盖路径 |
| `python3-devel` | pybind11 / nanobind 扩展需要的 `Python.h` |
| `gtest-devel` | C++ 单元测试（CTest） |
| `unzip` / `rpm`（提供 `rpm2cpio`） / `cpio` | KDNN RELEASE 模式解包 release zip；`build.sh` 默认 `--kdnn-source RELEASE`，在 CMake 之前检查这三个命令 |
| `openssl-devel` | JIT cache key 哈希（OpenSSL Crypto）；仅 `ANNC_ENABLE_TENSORFLOW_ADDON=ON` 时需要 |
| `protobuf-devel` / `protobuf-compiler` | 启用 TF 或 ONNX 前端时必需；`protoc` 用于校验与重新生成 proto 源码。ONNX 侧缺失即报错；TF 侧会先尝试自动下载 fallback protoc |

### 4. LLVM / MLIR 与 nlohmann/json

无需手动安装：CMake `FetchContent` 在 configure 时自动下载到 `third_party/llvm/` 与 `third_party/json/`，并作为 ANNC 构建的一部分从源码编译。

> `build.sh` 会在缺失时自动安装 `pybind11` / `nanobind`；传入 `--no-install-deps` 可关闭自动安装。TensorFlow 需自行安装。

## 验证

```shell
./build.sh --install-prefix "$HOME/ANNC" --build-type Debug   # 详细选项见「从源码构建」
export PATH="$HOME/ANNC/bin:$PATH"
which annc-opt annc-asm
```

能解析出可执行文件路径即安装成功。构建选项、测试与覆盖率见[从源码构建](../guides/build-from-source.md)。
