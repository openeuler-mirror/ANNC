# ANNC 内置 Kernel 编写指南

这份文档只讲当前仓里推荐的 hand-written builtin kernel 写法。

## 核心约定

1. builtin kernel 的内部实现统一写成：

```cpp
void impl(::annc::runtime::AnncThreadPoolCtx* ctx, typed_memref_args...);
```

2. `_mlir_ciface_*` wrapper 的 ABI 保持原有 memref 参数形状，不额外暴露 `ctx` 首参。
3. `ctx` 只承载宿主前端提供的线程池 / 并行能力。
4. `input/output` 这类真正的算子数据参数继续显式放在函数参数里，不要塞进 `ctx`。
5. spec 里直接写 wrapper body，通过 `getCurrentThreadPoolCtx()` 取 bridge ctx，再转给 `impl(ctx, ...)`。

## 最小例子

实现文件：`my_add_aarch64.cpp`

```cpp
#include "Kernel/threadpool/ThreadPoolContext.h"
#include "Kernel/MemRefTypes.h"

namespace {

void my_add_impl(::annc::runtime::AnncThreadPoolCtx* ctx,
                 AnncMemRef1DF32* output,
                 AnncMemRef1DF32* input) {
    (void)ctx;

    float* out = ANNC_MEMREF_DATA(*output);
    const float* in = ANNC_MEMREF_DATA(*input);
    const int64_t size = ANNC_MEMREF_SIZE_1D(*input);

    for (int64_t i = 0; i < size; ++i) {
        out[i] = in[i] + 1.0f;
    }
}

} // namespace
```

spec 文件：`my_add_kernel_specs.inc`

```cpp
ANNC_KERNEL(
    Name("MyAdd").Backend("aarch64"),
    (AnncMemRef1DF32* output,
     AnncMemRef1DF32* input),
    { my_add_impl(::annc::runtime::getCurrentThreadPoolCtx(), output, input); })
```

这表示：

- 算子名是 `MyAdd`
- backend 是 `aarch64`
- wrapper ABI 是 `(output, input)`
- wrapper 会先从 bridge TLS 取当前 ctx，再转给 `my_add_impl(ctx, output, input)`

## `AnncThreadPoolCtx` 里有什么

当前只保留一层数据结构：

```cpp
struct AnncThreadPoolCtx {
    void* impl;

    int (*num_threads)(void* impl);
    bool (*in_parallel_region)(void* impl);
    void (*parallel_for)(...);
};
```

它表达的是“宿主前端提供给这次 kernel 调用的并行能力句柄”，不是线程池对象本体，也不是通用大 runtime ctx。

## 推荐写法

默认 kernel：

```cpp
void my_kernel_impl(::annc::runtime::AnncThreadPoolCtx* ctx,
                    AnncMemRef1DF32* output,
                    AnncMemRef1DF32* input);
```

spec：

```cpp
ANNC_KERNEL(
    Name("MyKernel").Backend("aarch64"),
    (AnncMemRef1DF32* output,
     AnncMemRef1DF32* input),
    { my_kernel_impl(::annc::runtime::getCurrentThreadPoolCtx(), output, input); })
```

带类型约束的 kernel：

```cpp
template <typename T>
void my_scale_impl(::annc::runtime::AnncThreadPoolCtx* ctx,
                   AnncMemRef* output,
                   AnncMemRef* input);

ANNC_KERNEL(
    Name("MyScale")
        .Backend("aarch64")
        .TypeConstraint<float>("Tdata"),
    (AnncMemRef* output, AnncMemRef* input),
    { my_scale_impl<float>(::annc::runtime::getCurrentThreadPoolCtx(), output, input); })
```

## wrapper 什么时候写复杂一点

如果 wrapper 不只是简单转调，例如还需要：

- 参数重排
- 额外检查
- 预处理 / 后处理

那就直接在 body 里展开，例如：

```cpp
ANNC_KERNEL(
    Name("MyKernel").Backend("aarch64"),
    (AnncMemRef* output, AnncMemRef* input),
    {
        // extra logic
        auto* ctx = ::annc::runtime::getCurrentThreadPoolCtx();
        my_kernel_impl(ctx, output, input);
    })
```

## 新增 kernel 的步骤

1. 在 `builtin_kernels/` 下新增 `xxx_aarch64.cpp`
2. 在同目录新增 `xxx_kernel_specs.inc`
3. 重新构建

构建系统会自动生成：

- wrapper 源文件
- builtin registry 源文件
- `_mlir_ciface_*` 导出符号

## 你不需要做的事

你不需要：

- 手写 symbol 字符串
- 手写 wrapper 函数名
- 手写 registry 代码
- 改 `kernel_res.cpp`

## 当前目录里的参考实现

- 默认 kernel：[matmul_aarch64.cpp](/home/doe/ANNC-next/annc/lib/Kernel/builtin_kernels/matmul_aarch64.cpp)
- 默认 kernel spec：[matmul_kernel_specs.inc](/home/doe/ANNC-next/annc/lib/Kernel/builtin_kernels/matmul_kernel_specs.inc)
- 类型特化 kernel：[typed_scale_aarch64.cpp](/home/doe/ANNC-next/annc/lib/Kernel/builtin_kernels/typed_scale_aarch64.cpp)
- 类型特化 spec：[typed_scale_kernel_specs.inc](/home/doe/ANNC-next/annc/lib/Kernel/builtin_kernels/typed_scale_kernel_specs.inc)
