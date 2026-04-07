# ANNC 内置 Kernel 编写指南

这份文档只讲一件事：怎么在 `annc/lib/Kernel/builtin_kernels` 里新增一个 hand-written kernel。

你只需要写两个文件：

1. 实现文件：`xxx_aarch64.cpp`
2. spec 文件：`xxx_kernel_specs.inc`

其余事情都由构建系统自动完成：

- 自动生成 wrapper
- 自动生成 builtin registry
- 自动生成最终 `_mlir_ciface_*` symbol

## 一眼看懂的最小例子

### 1. 写实现文件

文件：`my_add_aarch64.cpp`

```cpp
#include "Kernel/MemRefTypes.h"

namespace {

void my_add_impl(AnncMemRef1DF32* output, AnncMemRef1DF32* input) {
    float* out = ANNC_MEMREF_DATA(*output);
    const float* in = ANNC_MEMREF_DATA(*input);
    const int64_t size = ANNC_MEMREF_SIZE_1D(*input);

    for (int64_t i = 0; i < size; ++i) {
        out[i] = in[i] + 1.0f;
    }
}

} // namespace
```

### 2. 写 spec 文件

文件：`my_add_kernel_specs.inc`

```cpp
ANNC_KERNEL(
    Name("MyAdd").Backend("aarch64"),
    ANNC_FORWARD(
        my_add_impl,
        (AnncMemRef1DF32* output, AnncMemRef1DF32* input),
        (output, input)))
```

这表示：

- 算子名是 `MyAdd`
- 后端是 `aarch64`
- wrapper 参数是 `(output, input)`
- wrapper 只是把参数直接转调给 `my_add_impl(output, input)`

## 你真正需要记住的写法

统一只用一个接口：

```cpp
ANNC_KERNEL(
    Name("算子名")
        .Backend("aarch64")
        .TypeConstraint<类型>("类型参数名"),
    ANNC_FORWARD(
        实现函数,
        (wrapper 参数列表),
        (调用实现函数时的参数列表)))
```

说明：

- 第 1 个参数是链式 builder，风格和 TensorFlow 接近
- `TypeConstraint<...>(...)` 是可选的
- `ANNC_FORWARD(...)` 表示“wrapper 只是转调实现函数”
- builtin kernel wrapper 固定为 `void` 返回，所以你不需要写 `void`

## 默认 kernel 怎么写

大多数固定类型算子都属于这种场景。

实现文件：

```cpp
#include "Kernel/MemRefTypes.h"

namespace {

void my_kernel_impl(AnncMemRef1DF32* output, AnncMemRef1DF32* input) {
    // compute
}

} // namespace
```

spec 文件：

```cpp
ANNC_KERNEL(
    Name("MyKernel").Backend("aarch64"),
    ANNC_FORWARD(
        my_kernel_impl,
        (AnncMemRef1DF32* output, AnncMemRef1DF32* input),
        (output, input)))
```

## 带类型约束的 kernel 怎么写

只有当算子 schema 里真的有动态类型参数时，才需要写 `TypeConstraint`。

例如 TensorFlow 里这种：

- `Input("input0: Tindices1")`
- `Attr("Tindices1: {int32, int64}")`

这时 spec 可以写成：

```cpp
ANNC_KERNEL(
    Name("MyGather")
        .Backend("aarch64")
        .TypeConstraint<int64_t>("Tindices1")
        .TypeConstraint<int32_t>("Tindices2"),
    ANNC_FORWARD(
        my_gather_impl<int64_t, int32_t>,
        (AnncMemRef* output, AnncMemRef* input0, AnncMemRef* input1),
        (output, input0, input1)))
```

## 模板实现的常见写法

实现文件：

```cpp
#include "Kernel/MemRefTypes.h"

namespace {

template <typename T>
void my_scale_impl(AnncMemRef* output, AnncMemRef* input) {
    // compute
}

} // namespace
```

spec 文件：

```cpp
ANNC_KERNEL(
    Name("MyScale")
        .Backend("aarch64")
        .TypeConstraint<float>("Tdata"),
    ANNC_FORWARD(
        my_scale_impl<float>,
        (AnncMemRef* output, AnncMemRef* input),
        (output, input)))

ANNC_KERNEL(
    Name("MyScale")
        .Backend("aarch64")
        .TypeConstraint<int32_t>("Tdata"),
    ANNC_FORWARD(
        my_scale_impl<int32_t>,
        (AnncMemRef* output, AnncMemRef* input),
        (output, input)))
```

## 什么场景不适合直接用 `ANNC_FORWARD`

如果 wrapper 不只是简单转调实现函数，而是还需要额外逻辑，例如：

- 参数重排
- 预处理/后处理
- 额外断言或分支

那就直接写完整 body：

```cpp
ANNC_KERNEL(
    Name("MyKernel").Backend("aarch64"),
    (AnncMemRef* output, AnncMemRef* input),
    {
        // extra logic
        my_kernel_impl(output, input);
    })
```

## 新增 kernel 的步骤

1. 在 `builtin_kernels/` 下新增 `xxx_aarch64.cpp`
2. 在同目录新增 `xxx_kernel_specs.inc`
3. 重新构建

只要文件名符合约定，构建系统会自动发现并生成：

- wrapper 源文件
- builtin registry 源文件

## 你不需要做的事

你不需要：

- 手写 symbol 字符串
- 手写 wrapper 函数
- 手写 registry 代码
- include `BuiltinKernelSpecDefine.h`
- include `BuiltinKernelSpecRegister.h`
- 改 `kernel_res.cpp`

## 注意事项

1. spec 文件名必须是 `*_kernel_specs.inc`
2. 实现文件名必须和 spec 对应，例如：
   - `typed_scale_aarch64.cpp`
   - `typed_scale_kernel_specs.inc`
3. `ANNC_KERNEL(...)` 里的实现函数必须能在对应实现文件里找到
4. `TypeConstraint` 只有在算子 schema 存在动态类型属性时才需要写
5. 一个 specialization 写一个 `ANNC_KERNEL(...)`

## 当前目录里的参考实现

- 默认 kernel： [matmul_aarch64.cpp](/home/doe/ANNC-next/annc/lib/Kernel/builtin_kernels/matmul_aarch64.cpp)
- 默认 kernel spec： [matmul_kernel_specs.inc](/home/doe/ANNC-next/annc/lib/Kernel/builtin_kernels/matmul_kernel_specs.inc)
- specialization kernel： [typed_scale_aarch64.cpp](/home/doe/ANNC-next/annc/lib/Kernel/builtin_kernels/typed_scale_aarch64.cpp)
- specialization spec： [typed_scale_kernel_specs.inc](/home/doe/ANNC-next/annc/lib/Kernel/builtin_kernels/typed_scale_kernel_specs.inc)
