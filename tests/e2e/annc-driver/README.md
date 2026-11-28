# 手工 driver / demo 资产

这些文件原先放在 `tests/annc/`，但没有任何 CMake、脚本或测试引用，属于历史
手工调试资产。迁移后集中放在这里，**不接入 CTest**：

| 文件 | 用途 |
|---|---|
| `demo.mlir` | `annc` driver 的 matmul+bias 手工输入 |
| `demo2.mlir` | `annc` driver 的 matmul_add_relu 手工输入 |
| `driver_static.c` | 静态链接调用 compiled kernel 的 ABI 示例 |
| `driver_dynamic.c` | `dlopen` 调用 compiled kernel 的 ABI 示例（可用 `ANNC_LIBRARY_NAME` 指定库） |
| `annc_threadpool_dynamic_driver.cpp` | 动态加载 kernel 并注入线程池的手工 driver |

使用方式（示例）：

```bash
# 静态驱动：demo2.mlir 的导出函数与 driver_static.c 的 extern 符号一致
build/bin/annc tests/e2e/annc-driver/demo2.mlir \
  -t tests/e2e/annc-driver/driver_static.c -o /tmp/driver_static

# 动态驱动：`--shared` 与 `-t` 同时使用时 `-o` 会被共享库和测试可执行文件共用，
# 因此先由 annc 生成 .so，再单独编译驱动。库文件名须与 MLIR 导出符号同名，
# 并以 ./ 开头引用，否则 dlopen 不搜索当前目录、driver 也无法派生符号。
build/bin/annc tests/e2e/annc-driver/demo.mlir --shared -o fused_matmul.so
clang -O3 tests/e2e/annc-driver/driver_dynamic.c \
  -DM=1536 -DK=1344 -DN=1152 -o /tmp/driver_dynamic -ldl -lm
ANNC_LIBRARY_NAME=./fused_matmul.so /tmp/driver_dynamic
```

如果后续要把其中任何一个变成自动测试，应把 driver、输入和断言放到对应的
`tests/e2e/` 子目录，并在 `tests/CMakeLists.txt` 或 `tests/integration/*/CMakeLists.txt`
注册；不要重新放回 `tests/unit/` 或生产目录。
