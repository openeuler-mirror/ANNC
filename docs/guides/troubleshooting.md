# 常见问题

## 构建相关

1. **编译失败**：确认已安装所有依赖（pybind11、nanobind、TensorFlow、cmake、ninja-build），见[安装 ANNC](../getting-started/installation.md)。
2. **编译内存不足**：减少并行编译任务数，如 `ninja -j4`。
3. **LLVM 下载缓慢**：Gitee 镜像偶尔不稳定，可重试或手动下载 LLVM 源码到 `third_party/llvm/`。

## 运行相关

4. **`annc` 链接失败**：确认 `ANNC_CLANG` 指向可用的 clang，且 `ANNC_KERNEL_LIB_PATH` 指向 `ninja install` 的安装树 `lib` 目录（`annc` 需要该目录下的 kernel 归档与 LLVM 归档）。
5. **Python 绑定导入失败**：确认已安装 `pybind11` 和 `nanobind`，且构建时使用了正确的 Python 解释器（`build.sh` 读取 `PYTHON`，并传递给 `Python_EXECUTABLE` / `Python3_EXECUTABLE`）。

更多环境变量见[环境变量参考](environment-variables.md)，构建选项见[从源码构建](build-from-source.md)。
