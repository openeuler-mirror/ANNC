# KernelContext 基础设计

状态：本说明对应 Context 基础定义与公共封装。

## 1. 目标与职责边界

统一 kernel 的外部调用入口，将计算参数与宿主能力通过一个 Context 传递：

```cpp
extern "C" AnncStatusCode annc_kernel_<symbol>(AnncKernelContext* context);
```

kernel 负责计算与动态输出形状；Context 提供框架无关的 tensor 表示和服务协议；
宿主负责参数绑定、内存所有权、线程池和执行状态。公共层不依赖 TensorFlow、
ONNX Runtime、LLVM/MLIR、kernel registry 或具体线程池实现。

```text
框架 tensor / 站点参数
  → 宿主适配层：绑定输入、输出、属性与服务
  → AnncKernelContext：稳定的 C 调用边界
  → KernelContext：kernel 使用的轻量 C++ 封装
  → 算法；需要分配或调度时回调宿主
```

Context 不承担图执行、算子选择或参数排列推导。编译期 schema 约定输入、输出
和属性的槽位、类型与 rank，融合/调用侧按该约定绑定，kernel 按相同索引访问。

## 2. 文件与分层

| 文件 | 职责 |
| --- | --- |
| [KernelContext.h](include/Context/KernelContext.h) | C ABI：状态码、tensor、属性、调用上下文和运行时服务 |
| [KernelContextUtils.h](include/Context/KernelContextUtils.h) | C++ 访问、分配、调度和入口封装 |
| [StringRef.h](include/Context/StringRef.h) | 借用字符串视图，兼供旧 MemRef 表示使用 |
| [ThreadPool.h](include/Context/ThreadPool.h) | 宿主线程池抽象、调度选项及 TLS 访问声明 |
| [Parallel.h](include/Context/Parallel.h) | 通用同步调度与生成代码桥接声明 |
| [RowParallel.h](include/Context/RowParallel.h) | 行任务的成本估算、分片粒度与调度选项封装 |
| [ThreadPool.cpp](lib/ThreadPool.cpp)、[Parallel.cpp](lib/Parallel.cpp) | TLS 与同步调度桥接，不创建线程 |
| [CMakeLists.txt](CMakeLists.txt) | 头文件目标 `ANNCContext`、静态桥接库 `ANNCThreadPool` 和安装规则 |

CMake 使用方链接 `ANNCContext`；使用 TLS/生成代码桥接时另链接 `ANNCThreadPool`。
手动编译公共 Kernel 头文件时同时增加 `-I<仓库>/annc/include` 和
`-I<仓库>/context/include`；安装后两组头文件统一位于安装前缀的 `include/` 下。
`ANNCContext` 不生成动态库，`ANNCThreadPool` 是静态库。

C ABI 只暴露普通结构、指针和函数指针；STL、框架对象与线程池虚表不跨越此边界。
C++ `KernelContext` 借用 `AnncKernelContext*`，不复制 tensor 或持有执行资源。
旧 `AnncExecutionContext` 继续引用公共状态码与分配描述，但布局不变，
不能与新 `AnncKernelContext` 互相强转。

## 3. 一次调用包含的数据

`AnncKernelContext` 保存 `struct_size`、`abi_version`，以及以下数据：

| 字段 | 含义 |
| --- | --- |
| `inputs` / `num_inputs` | 输入 tensor 视图及数量，支持可变输入 |
| `outputs` / `num_outputs` | 语义输出槽位及数量，槽位可已绑定或尚未分配 |
| `attrs` / `num_attrs` | 独立的属性槽位及数量 |
| `services` | 本次执行的宿主服务表 |

输入、输出、属性各自从索引 0 开始，彼此不共享索引空间。
ABI 名称为 `annc_context_v3`，版本为 3；

### 3.1 Tensor 视图与分配请求

`AnncTensorView` 借用数据地址和布局元数据：

| 字段 | 含义 |
| --- | --- |
| `allocated` / `aligned` | 原始分配地址 / 寻址基地址 |
| `offset` | 首个逻辑元素相对 `aligned` 的偏移，以元素为单位 |
| `sizes` / `strides` | 各维大小与步长，步长以元素为单位 |
| `element_type` / `rank` | 元素类型和维数 |
| `layout` / `flags` | 布局，以及已绑定、只读等标志 |

当前定义 F32、I32、I64、STRING，辅助层支持 rank 0–8、非负 offset/strides。
连续 `2 × 3` tensor 的 sizes 为 `[2, 3]`、strides 为 `[3, 1]`。
rank 0 表示一个标量元素；空 tensor 可没有数据地址，仍须用 BOUND 标志表示绑定。
`contextData<T>(view)` 根据 `aligned` 和 `offset` 获取指针，不转换数值类型。

`AnncTensorDesc` 表达分配请求，包含类型、rank、dims、flags 等信息；
`AnncTensorView` 表达分配得到或预先绑定的实际 tensor，二者职责不同。
`AnncStringRef` 只包含字符串指针和长度，不拥有字符数据，也不暴露 TF 字符串类型。

### 3.2 标量属性

`AnncAttributeView` 包含 `kind`、保留字段 `reserved`、`data` 和 `count`。
当前仅定义 I64 标量属性，要求 `count == 1`、`reserved == 0`。
使用 `getI64Attr(slot)` 返回 `std::optional<int64_t>`，失败时记录错误。

属性与 tensor 标量不同：num_buckets 等配置放入 attrs；作为算子输入的零维
F32/I32/I64 tensor 仍放入 inputs。数组 tensor 不因其内容为整数就改成属性。
属性类别按实际需求扩展，不提前增加未使用的种类。

## 4. 宿主服务、所有权与同步约定

`AnncRuntimeServices` 的 `opaque` 指向宿主私有状态，回调通过它找到分配器、
框架执行上下文和错误记录；公共层不解释该对象的类型。

| 服务 | 用途 | 是否必需 |
| --- | --- | --- |
| `allocate_output` | 按局部输出槽位分配并返回视图 | 必需 |
| `allocate_temp` | 分配临时 tensor 并返回视图 | 必需 |
| `report_error` / `get_status` | 记录错误 / 查询执行状态 | 必需 |
| `parallel_for` / `num_threads` | 同步并行执行 / 查询线程数 | 必需 |
| `shard` | 传递最大并行度和单位工作成本 | 可选 |
| `native_thread_pool` | 按 provider 借用兼容的原生线程池 | 可选 |

通过版本和结构大小检查基础前缀，访问可选字段前检查 `struct_size` 是否覆盖该字段。
这允许同版本服务表追加可选尾字段，不允许任意改变已有字段布局或含义。

输入、输出、shape/strides、临时空间及字符串数据的所有权留在宿主。
分配请求的 dims 只在同步回调期间借用，宿主必须为返回视图保管有效的元数据。
Context 及其服务必须存活到同步调用和任务结束；kernel 不自行释放这些资源。
缺少 `parallel_for` 或 `num_threads` 的服务表会被拒绝；串行宿主也须提供串行回调。
shard 无可用的限并行度服务时串行执行以遵守限制。宿主回调只能借用框架池，
必须同步完成任务并在嵌套并行区域串行执行；公共层不创建备用线程池。
原生线程池不兼容或不可用时返回空句柄，由消费方选择现有替代执行路径。

## 5. kernel 侧 C++ 接口与检查边界

| 方法 | 行为 |
| --- | --- |
| `getInput(slot)` | 获取连续输入视图 |
| `getStridedInput(slot)` | 获取允许非连续布局的输入视图 |
| `getI64Attr(slot)` | 获取 I64 标量属性 |
| `numInputs()` | 查询输入数量 |
| `allocateOutput<T>(slot, dims, flags)` | 分配或复用输出，rank 从 dims 数量推导 |
| `allocateTemp<T, Rank>(dims, &view)` | 请求临时 tensor |
| `getThreadPool()` | 获取借用的调度句柄 |
| `error(code, message)` / `status()` | 上报错误并返回错误码 / 查询状态 |

`allocateOutput` 对未绑定槽位调用宿主分配，对已绑定槽位检查类型、shape、连续布局
和可写性后复用。它既用于动态形状输出，也用于预先分配的输出；没有单独的 getOutput。
flags 可省略，需清零时使用 `ANNC_ALLOCATION_ZERO`。

输入访问检查槽位、视图结构、布局与寻址算术；不会按 kernel schema 再验证期望
的 dtype/rank。固定契约由编译器和调用入口保证，动态索引边界、形状关系与算法溢出
仍由对应实现处理。视图检查没有底层分配容量信息，不能证明实际 buffer 容量足够。

`invokeContextKernel` 检查基础 ABI、创建 C++ view 并处理状态传播，
将调用线程捕获的 bad_alloc 和其他异常转换成错误码。它不替代宿主或工作线程内部
的异常边界。算法内部 SIMD、hash、GEMM helper 仍可使用显式参数。

## 6. 宿主适配与调用流程

TensorFlow 与 ONNX Runtime 的 builtin 调用共用 `AnncKernelContext`。
框架适配层将框架 tensor 包装成视图，并提供分配、错误和线程服务：

| 适配层 | 实现位置 | 职责 |
| --- | --- | --- |
| TensorFlow | `tensorflow_addons/context/` | 绑定 TF tensor，提供输出/临时分配与 TF 线程池服务 |
| ONNX Runtime | `addons/onnxruntime/annc_ep_execution.*` | 按站点契约绑定 ORT 输入输出，提供分配、调度及原生线程池句柄 |

一次同步调用按以下顺序执行：

1. 根据所选 kernel 的契约排列输入、输出和属性，检查入口参数。
2. 创建 tensor 视图、服务表及 Context，绑定本次执行的宿主状态。
3. 调用 kernel；由 kernel 按索引读取参数、请求输出并执行计算。
4. 检查返回码、宿主错误状态与输出绑定情况，向框架交付输出。
5. 调用结束后释放本次临时资源，输出继续由框架持有。

## 7. 线程池与行任务调度

共享线程池接口、行任务调度辅助和 TLS 桥接位于 `context/include/Context/` 和
`context/lib/`；所有调用方（包括生成式 GEMM）直接引用 `Context/ThreadPool.h`、
`Context/Parallel.h` 或 `Context/RowParallel.h`，不保留旧路径的转发头。

两类调用共用宿主调度能力，但入口不同：

- Context kernel：`getThreadPool()` 返回 `KernelThreadPool`，通过 C 服务表回调宿主，
  不需要读取 TLS，也不把 C++ 线程池虚表传入 Context ABI。
- 生成代码及已有 C++ 调用方：使用 `annc::threadpool::AnncThreadPool` 抽象。
  宿主设置当前线程的借用池，生成代码通过 `annc_threadpool_*` 桥接函数访问；
  相关调用方链接 `ANNCThreadPool`。移动这些公共头文件不改变生成式 GEMM 的参数 ABI。

对于 `annc::threadpool::parallel_for`，`ParallelForOptions::max_parallelism`
仅在公共辅助函数中作为串并行开关：值小于等于 1 时串行；大于 1 或未设置时允许并行
（仍受任务粒度、线程数和嵌套并行检查约束）。选项原样传给宿主的
`AnncThreadPool::parallel_for`，大于 1 的具体上限需要宿主实现支持；公共辅助函数
本身不保证该上限。因此，设置为 2 和 4 在未实现限流的宿主上可能使用相同的并行度。
下述 Context `shard` 接口则要求遵守并行度上限，缺少对应宿主服务时退回串行。

`KernelThreadPool` 提供以下方法：

| 方法 | 行为 |
| --- | --- |
| `numThreads()` | 查询线程数，非正值归一为 1 |
| `parallelFor(total, task)` | 同步处理 `[0, total)`，task 接收 `[begin, end)` |
| `shard(maxParallelism, total, costPerUnit, task)` | 带并行度上限和成本提示的同步分片 |
| `nativePool(provider)` | 按 provider 获取兼容的宿主原生池，不可用时返回空句柄 |

`annc::kernels::Shard(...)` 是 `shard` 的便捷函数。服务表缺少可选 `shard` 回调时，
带并行度上限的请求串行执行；`parallelFor` 的嵌套并行保护由宿主回调负责。
ORT 使用的原生池 provider 为 `"onnxruntime"`，句柄仅在本次同步调用期间有效。

`RowParallel.h` 只计算调度提示，不执行任务、不持有线程池：

- `cost::elementRow(...)`、`cost::dotRow(...)` 等估算单位行的工作成本。
- `grainForRowWork(...)` 按每行元素操作数估算分片粒度。
- `rowOptions(costPerUnit, elementsPerRow)` 将成本与粒度合并为 `ParallelForOptions`，
  供 `annc::threadpool::parallel_for` 使用。Context kernel 也可取成本值传给 `Shard`。
