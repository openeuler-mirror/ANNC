# 常见问题

## 构建相关

1. **编译失败**：确认已安装所有依赖（cmake、ninja-build、TensorFlow、24.03 的 `llvm-toolset-20-*` 或 22.03 的自建 LLVM/MLIR 20/21；`--enable-python` 时再加 pybind11/nanobind），见[安装 ANNC](../getting-started/installation.md)。
2. **编译内存不足**：减少并行编译任务数，如 `ninja -j4`。
3. **找不到 LLVM/MLIR**：configure 报 `LLVMConfig.cmake`/`MLIRConfig.cmake` 缺失。24.03 安装 `llvm-toolset-20-*`；22.03 仓库只有 LLVM/MLIR 12.0.1，需自建 20/21 后传 `-DLLVM_DIR`/`-DMLIR_DIR`。工具链在自定义前缀时导出 `ANNC_LLVM_BIN=<LLVM bin 目录>`，或传 `-DLLVM_DIR`/`-DMLIR_DIR`。标准 yum 安装且 `llvm-config-20` 在 `PATH` 时无需导出。

## 运行相关

4. **`annc` 链接失败**：确认 `ANNC_LLVM_BIN`（或构建期记录目录）下有 `clang` 与 `libLLVMSupport.a`/`libLLVMDemangle.a`；`ANNC_KERNEL_LIB_PATH` 只指向 ANNC 归档目录（aarch64 为 `ninja install` 的 `<prefix>/lib64`）。缺 `clang` 时安装 `llvm-toolset-20-clang`。
5. **Python 绑定导入失败**：Python 绑定默认不构建；确认构建时加了 `--enable-python`，并安装 `pybind11` 和 `nanobind`，且使用了正确的 Python 解释器（`build.sh` 读取 `PYTHON`，并传递给 `Python_EXECUTABLE` / `Python3_EXECUTABLE`）。

更多环境变量见[环境变量参考](environment-variables.md)，构建选项见[从源码构建](build-from-source.md)。
