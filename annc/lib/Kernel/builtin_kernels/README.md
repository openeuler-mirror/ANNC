# ANNC 手写 Kernel 编写指南

这份文档只讲“怎么写一个 hand-written builtin kernel”。

## 1. 你需要写什么

每个 kernel 通常对应两个文件：

1. `xxx_aarch64.cpp`
2. `xxx_kernel_specs.inc`

前者写实现，后者写注册和 wrapper 形状。

## 2. 实现函数怎么写

推荐形状：

```cpp
void impl(::annc::threadpool::AnncThreadPool* thread_pool,
          typed_memref_args...);
```

说明：

- `thread_pool` 是当前调用可用的线程池抽象
- 真正的输入输出数据仍然显式写在参数里
- 不要把算子数据塞进线程池对象

最小例子：

```cpp
#include "Kernel/threadpool/ThreadPool.h"
#include "Kernel/MemRefTypes.h"

namespace {

void my_add_impl(::annc::threadpool::AnncThreadPool* thread_pool,
                 AnncMemRef1DF32* output,
                 AnncMemRef1DF32* input) {
    (void)thread_pool;

    float* out = ANNC_MEMREF_DATA(*output);
    const float* in = ANNC_MEMREF_DATA(*input);
    const int64_t size = ANNC_MEMREF_SIZE_1D(*input);

    for (int64_t i = 0; i < size; ++i) {
        out[i] = in[i] + 1.0f;
    }
}

} // namespace
```

## 3. spec 怎么写

spec 负责两件事：

1. 注册逻辑算子名和 backend
2. 定义 `_mlir_ciface_*` wrapper 的参数形状

最小例子：

```cpp
ANNC_KERNEL(
    Name("MyAdd").Backend("aarch64"),
    (AnncMemRef1DF32* output,
     AnncMemRef1DF32* input),
    { my_add_impl(::annc::threadpool::getCurrentThreadPool(), output, input); })
```

这里的含义是：

- 算子名是 `MyAdd`
- backend 是 `aarch64`
- wrapper ABI 是 `(output, input)`
- wrapper 从 TLS bridge 取当前线程池，再转给实现函数

## 4. 什么时候用并行 helper

如果 kernel 需要并行，优先用：

```cpp
#include "Kernel/threadpool/Parallel.h"

annc::kernels::parallel_for(thread_pool, total, fn);
annc::kernels::parallel_for(thread_pool, total, options, fn);
```

最常用的是简洁版：

```cpp
annc::kernels::parallel_for(thread_pool, total,
                            [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
        ...
    }
});
```

如果需要调度提示，再传 `options`：

```cpp
annc::threadpool::ParallelForOptions options;
options.grain_size = 16;

annc::kernels::parallel_for(thread_pool, total, options, fn);
```

## 5. 带类型约束的写法

```cpp
template <typename T>
void my_scale_impl(::annc::threadpool::AnncThreadPool* thread_pool,
                   AnncMemRef* output,
                   AnncMemRef* input);

ANNC_KERNEL(
    Name("MyScale")
        .Backend("aarch64")
        .TypeConstraint<float>("Tdata"),
    (AnncMemRef* output, AnncMemRef* input),
    { my_scale_impl<float>(::annc::threadpool::getCurrentThreadPool(), output, input); })
```

## 6. wrapper 什么时候写复杂一点

如果 wrapper 不只是简单转调，比如还需要：

- 参数重排
- 额外检查
- 预处理 / 后处理

就直接在 body 里展开：

```cpp
ANNC_KERNEL(
    Name("MyKernel").Backend("aarch64"),
    (AnncMemRef* output, AnncMemRef* input),
    {
        auto* thread_pool = ::annc::threadpool::getCurrentThreadPool();
        my_kernel_impl(thread_pool, output, input);
    })
```

## 7. 新增 kernel 的步骤

1. 在 `builtin_kernels/` 下新增 `xxx_aarch64.cpp`
2. 在同目录新增 `xxx_kernel_specs.inc`
3. 重新构建

## 8. 参考实现

- 默认 kernel：[`matmul_aarch64.cpp`](matmul_aarch64.cpp)
- 默认 kernel spec：[`matmul_kernel_specs.inc`](matmul_kernel_specs.inc)
