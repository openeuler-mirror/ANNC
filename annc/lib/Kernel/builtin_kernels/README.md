# ANNC 手写 Kernel 编写指南

这份文档只讲“怎么写一个 hand-written builtin kernel”。

## 1. 确定契约与复用范围

先检索已有实现、共享 schema 和融合模式。为已有算子新增 backend 或类型特化时复用原契约；只有新增算子或边界变化时才扩展 schema 和匹配逻辑。

| 需要确定的内容 | 约定位置 |
| --- | --- |
| 算子名称、输入/输出顺序、rank、类型变量与整数属性 | [BuiltinKernelSchema.cpp](../../Dialect/Atir/BuiltinKernelSchema.cpp) 中的 getContextKernelSchema |
| backend、支持的类型与 ABI | 本目录的 *_kernel_specs.inc |
| 子图语义、形状约束与融合边界 | [OpFusion.cpp](../../Dialect/Atir/Passes/OpFusion.cpp)、[CustomPatterns/](../../Dialect/Atir/Passes/Patterns/CustomPatterns) 或已有配置融合规则 |
| 数值语义、动态输出形状、错误边界 | kernel 实现及对应测试 |

输入、输出和整数属性分别从槽位 0 编号，各自按 schema 声明顺序传递。属性槽位独立于张量槽位，输出槽位不由分配先后决定。例如 Gather 的 unique_values、unique_indices、gathered 固定对应输出槽 0、1、2。运行时 Context 不携带 schema 或参数名称。

## 2. 添加实现与 spec 文件

同一目录下按以下命名配对；一个 spec 文件可声明多个类型或变体：

```text
foo_aarch64.cpp
foo_kernel_specs.inc
```

[CMakeLists.txt](CMakeLists.txt) 递归发现 spec，按其文件名查找对应的 foo_aarch64.cpp，生成 wrapper 和注册代码。实现由生成的 wrapper 包含，不需要再手写导出函数或重复注册，也不要修改 *.gen.cpp 生成物。

参考 [matmul_aarch64.cpp](matmul_aarch64.cpp) 和 [matmul_kernel_specs.inc](matmul_kernel_specs.inc)。KDNN 实现放在 kdnn_adaptor/，由 ANNC_ENABLE_KDNN_ADAPTOR 控制；packed 变体还受 ANNC_KDNN_HAS_PACKED_B 门控。新增外部依赖时须同步处理构建和链接规则。

## 3. 编写 Context 实现

下面以已有 MatMul 契约说明实现方式。示例执行连续 F32 矩阵乘法，可作为阅读骨架；仓库已有 MatMul，不需要重复添加。实际实现的分片调度见 [matmul_aarch64.cpp](matmul_aarch64.cpp)。

```cpp
#include "Context/KernelContextUtils.h"

namespace {

AnncStatusCode matmul_example_impl(annc::kernels::KernelContext& context) {
  const auto* lhs = context.getInput(0);
  const auto* rhs = context.getInput(1);
  if (!lhs || !rhs) return context.status();
  if (lhs->sizes[1] != rhs->sizes[0]) {
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "MatMul inner dimensions must match");
  }

  const int64_t m = lhs->sizes[0];
  const int64_t k = lhs->sizes[1];
  const int64_t n = rhs->sizes[1];
  auto* output = context.allocateOutput<float>(0, {m, n});
  if (!output) return context.status();

  const float* a = annc::kernels::contextData<float>(*lhs);
  const float* b = annc::kernels::contextData<float>(*rhs);
  float* out = annc::kernels::contextData<float>(*output);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int64_t p = 0; p < k; ++p) sum += a[i * k + p] * b[p * n + j];
      out[i * n + j] = sum;
    }
  }
  return context.status();
}

}  // namespace
```

编写实现时遵循以下规则：

- `getInput(index)` 校验索引、view 有效性和连续布局。需要处理非连续输入时使用 `getStridedInput(index)`，计算中正确处理 strides/offset。固定 dtype/rank 由编译期 schema 与宿主入口保证，访问 helper 不替代这些契约。
- 保留算法所需的运行时检查，例如维度关系、索引越界与大小计算溢出。访问或分配失败后立即返回状态；可用 `context.error(code, message)` 记录诊断。
- 通过 `contextData<T>(view)` 获取计入 offset 的指针。字符串输入是借用的 AnncStringRef 数组，由 TF adapter 转换，kernel 不直接使用 TF 的 tstring。
- KDNN/packed 实现复用已有适配器和格式契约。packed 数据不能作为普通 dense RHS 交给不支持该格式的后端。

常用接口如下，完整定义见 [KernelContextUtils.h](../../../../context/include/Context/KernelContextUtils.h)：

| 操作 | 接口与使用要求 |
| --- | --- |
| 分配输出 | `allocateOutput<T>(slot, {dims...}, flags)`：复用匹配的预绑定输出或回调宿主分配，返回 view，失败返回 nullptr。标量用 {}，flags 默认 ANNC_ALLOCATION_NONE |
| 临时空间 | `allocateTemp<T, Rank>(dims, &view)`：检查返回码，空间仅在本次同步调用有效 |
| 整数属性 | `getI64Attr(index)`：返回 optional，使用前检查；编译器及宿主通过 context_i64_attrs 传递 |
| 实例资源 | `getResource(type, version, bytes)`：按资源协议读取借用的只读 payload；可选资源缺失时按该特性的约定处理 |
| 线程调度 | `getThreadPool()` 取得借用句柄，使用 workers.parallelFor；或引入 Context/RowParallel.h 后调用 Shard(maxParallelism, workers, total, costPerUnit, fn)，cost 为每单位工作成本估计 |
| 错误传播 | `error(code, message)` 记录错误，`status()` 查询宿主状态；调度等接口返回的失败码也须向上传播 |

输出应通过分配 helper 绑定到对应槽位，不能替换宿主返回的数据或形状指针。线程调度同步完成；parallel_for/num_threads 是必需服务，串行宿主提供串行实现。可选 shard 服务缺失时 Shard 串行执行。kernel 不创建备用线程池，不保存超出调用生命周期的 Context、临时空间或线程池句柄。

## 4. 声明并注册 kernel

与上例配套的 spec 如下；实际新增时换成自己的算子名称与类型约束：

```cpp
ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_MatMulExample_float,
    Name("MatMul").Backend("aarch64").TypeConstraint<float>("T")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(context,
                                                 matmul_example_impl);
    })
```

spec token 用于生成符号，应避免与已有特化重复。Name 必须与 compiler schema/callee 名称一致，TypeConstraint 的变量名必须与 schema 一致，ABI 必须显式设置：registry 的默认 ABI 是 mlir_ciface。上例沿用已有 MatMul 注册条件，不能作为另一个独立实现直接加入生产构建。

invokeContextKernel 校验 Context 协议并将异常转换为状态码。CMake 通过 [BuiltinKernelWrapper.cpp.in](BuiltinKernelWrapper.cpp.in) 和共享 spec 宏生成声明、定义及注册；内部微内核或 vendor API 可继续沿用其调用约定。

## 5. 接入融合与 lowering

完成注册后，还需要让编译器识别并调用该 kernel：

1. 新算子在 [BuiltinKernelSchema.cpp](../../Dialect/Atir/BuiltinKernelSchema.cpp) 增加共享 schema；已有算子新增 backend/类型特化时复用原 schema。静态整数属性通过 I64AttrArg 声明，按声明顺序生成 context_i64_attrs。
2. 复用或扩展对应的融合 matcher / 配置规则，按 schema 规范顺序收集输入和边界输出，并标记 abi=annc_context_v3。仅在子图语义、类型、rank 和边界满足契约时匹配。
3. 使用现有 FastCodegen 与 --lower-context-kernel-entries 路径解析注册符号并生成单 Context 调用。该入口 pass 校验单 builtin 的融合边界；多 builtin 组合不能直接套用此入口。实现见 [LowerContextKernelEntries.cpp](../../Conversion/Context/LowerContextKernelEntries.cpp)。
4. 接入真实宿主时检查 TF 节点元数据或 ORT 站点报告、EP 支持范围与输入/输出绑定。注册 kernel 不会自动使新的融合模式进入 EP 接管范围。

普通生成代码仍有独立的 mlir_ciface 路径，不能仅修改 ABI 名称或函数指针类型来迁移旧产物。新增公共服务或修改 ABI 属于跨层契约变更，须同时评估 compiler、TF、ORT 和 kernel。

## 6. 构建与验证

在已按所需 backend 配置的构建目录中构建：

```bash
cmake --build build --target ANNCBuiltinKernels
```

该命令验证 kernel 库的构建。涉及 matcher、schema 或 lowering 时，还需构建对应的 annc-opt/annc-asm/pipeline 目标；配置见 [从源码构建](../../../../docs/guides/build-from-source.md)。

按改动范围补充并执行测试：

| 验证层次 | 放置位置与检查内容 |
| --- | --- |
| 数值与服务契约 | [tests/unit/kernels/](../../../../tests/unit/kernels)：执行实际 kernel，对比完整输出，覆盖相关的动态输出、多输出、属性及错误分支 |
| 注册与类型选择 | [builtin_shared_spec_test.cpp](../../../../tests/unit/kernels/builtin_shared_spec_test.cpp) 等：检查名称、backend、类型和 ABI 能解析到正确符号 |
| 融合与 lowering | [tests/FileCheck/](../../../../tests/FileCheck)：匹配正例、语义/边界不符的反例，以及生成的 Context 入口与调用 |
| 框架端到端 | [tests/e2e/kernel_context/](../../../../tests/e2e/kernel_context)：按 kernel 新增或扩展 TF/ORT 用例，对比原图与融合执行结果 |

通用 aarch64 Context 数值驱动可独立运行，只需 Python 与 C++17 编译器：

```bash
python3 -I tests/unit/kernels/run_context_kernels.py
ANNC_CONTEXT_SANITIZE=1 python3 -I tests/unit/kernels/run_context_kernels.py
```

runner 会编译顶层 Context spec，但数值用例在 [kernel_context_driver.cpp](../../../../tests/unit/kernels/kernel_context_driver.cpp) 中显式编写。新增 spec 不会自动产生数值覆盖，需要补充调用与期望结果；该 runner 不递归覆盖 KDNN 子目录，也不等于真实 TF/ORT 执行验证。

相关套件入口：

```bash
bash tests/FileCheck/run.sh Dialect/Atir
bash tests/FileCheck/run.sh Conversion
ctest --test-dir build -R KernelContext.ProductionKernels --output-on-failure
ANNC_REQUIRE_E2E=1 python3 -m pytest tests/e2e/kernel_context/ -v
```

FileCheck 依赖已构建工具与 LLVM 工具链；端到端用例依赖目标平台、框架及配套产物。KDNN、ORT/MLAS 分支须在对应依赖与平台上验证；缺依赖时跳过不构成通过。前置条件与注册项见 [测试说明](../../../../tests/README.md)，交付时分别报告实际执行、失败和跳过的检查。
