# addons/onnxruntime — ANNC 接入 ONNX Runtime

本目录是与 `tensorflow_addons/`（TF 侧接入）对称的 ONNX Runtime 侧接入实现：一个
对 ORT 头文件编译的独立共享库（EP 插件），**默认不参与 ANNC 根构建**：默认
`./build.sh` 不查找 ORT SDK、也不改变既有构建行为。

需要把 EP 插件纳入根构建时用可选开关：

```shell
export ANNC_ONNXRUNTIME_INCLUDE_DIR=<ORT_ROOT>/include/onnxruntime/core/session
./build.sh --enable-ort-ep          # 需同时启用 ONNX 前端（默认即启用）
```

对应 CMake 选项 `ANNC_ENABLE_ONNXRUNTIME_EP=ON` 与 `ANNC_ONNXRUNTIME_INCLUDE_DIR=<path>`。
该路径必须实际含有 `onnxruntime_c_api.h` 与 `onnxruntime_ep_c_api.h`，缺失时 configure
直接失败（不自动下载 SDK）；文件存在只验证路径，不代表版本兼容。开启后产出
`build/lib/libannc_ep.so` 并安装到 `<prefix>/<libdir>/libannc_ep.so`。**EP 编译与宿主加载验收
须使用 ORT 1.26.0 SDK/runtime**。

> 与 `tensorflow_addons/` 的差异：TF addon 由 `ANNC_ENABLE_TENSORFLOW_ADDON` 控制且默认
> 构建，本插件的开关 `ANNC_ENABLE_ONNXRUNTIME_EP` 默认关闭。

> 状态：**运行时编译是唯一形态**（2026-09-29 起）。当前 EP 接受两类站点：
> `abi=mlir_ciface`（固定 3D lhs + 2D rhs + 3D output 的预分配输出调用）与
> `abi=annc_execution_v2`（首参为执行上下文，输出经回调分配，线程池随上下文传入，
> 见下节）。context-ABI 站点覆盖 `tensordot_matmul`、`masked_mul`、`layer_norm` 和
> `batched_matmul_add`；其 pattern、成员节点和输出集合必须由 ANNC pipeline 报告，EP
> 只做契约校验和整体申报。

## 会话初始化时发生什么

```
session init
└─ GetCapabilityRuntimeCompileImpl(graph)
     ├─ 控制流子图（Graph_GetParentNode != null）→ 不申报
     ├─ 导出：OrtGraphToProto → model.onnx（自命名 n<i>/t<i>_<p>，initializer 内联）
     ├─ 签名 → 查磁盘缓存
     │    未命中 → annc-onnx-pipeline 现场编译（flock 单飞）→ sites.json + kernel .so
     │    任一失败 → 记 stderr、不申报任何节点、模型回落 CPU EP
     ├─ dlopen/dlsym 逐站点预校验（在申报之前，使 Compile 不可能因 kernel 解析失败）
     └─ 逐站点 AddNodesToFuse
└─ CompileImpl → 会话站点表取 kernel 名 → 填 ComputeInfo（并核对输出数/契约）
└─ ComputeImpl → 按 site.abi 分发：
     mlir_ciface        → OrtKernelContext → memref → _mlir_ciface_<kernel_name>
     annc_execution_v2  → 构造执行上下文（回调 + 线程池）→
                          _mlir_ciface_<kernel_name>(ctx, args...)
```

## annc_execution_v2：ONNX 执行上下文

上下文封装与 `tensorflow_addons/annc_fused_op.cc` 的 `TensorFlowExecutionState`
一一对应（输出/临时/错误三类回调），差别是**线程池作为执行上下文成员传入**
（ANNC ABI v3 的 `thread_pool` 字段），不再走宿主 TLS 注入。TF 与 ONNX 共用这
一个上下文抽象：TF 侧构造的上下文 `thread_pool` 为空（其池仍经
`annc_set_current_threadpool` 注入 kernel 库 TLS，kernel 从 TLS 取），kernel 侧
用 `resolveThreadPool(execution)` 取池——上下文池优先、为空回退 TLS。实现见
`annc_ep_execution.h/.cc`：

| 上下文成员 | ORT 实现 | 约束 |
|---|---|---|
| `allocate_output(handle, slot, desc)` | `KernelContext_GetOutput` + 张量类型复核 | slot 必须在站点声明的输出数内、不得重复分配；rank 与站点契约一致；元素类型与图声明一致（否则记录错误并拒绝） |
| `allocate_temp(handle, desc)` | `KernelContext_GetAllocator` + `AllocatorAlloc`（`ANNC_ALLOCATION_ZERO` 置零） | 逐 Compute 分配、Compute 结束 `AllocatorFree`；allocator 句柄最后 `ReleaseAllocator` |
| `report_error(handle, status, msg)` | 记录首个错误，Compute 返回时转 `OrtStatus` | 不会 terminate 宿主；失败即该次 Run 失败，不静默 |
| `thread_pool` | `OrtThreadPool`（原生池，宿主版本不匹配时回落公开 API） | 原生路径按 cost/grain 派发；公开路径每 shard 回调一次并映射为区间；公开调用失败回退本线程；嵌套由 thread_local 深度判定 |

调用约定（kernel 侧 spec 见 `annc/lib/Kernel/builtin_kernels/*_kernel_specs.inc`）：

- `site.args` 是 kernel 的**输入**参数表（按调用顺序）。EP 优先使用报告中的张量名绑定；
  名称不可用时才按 **rank + dtype** 解析，并对二义性 fail-closed。融合节点的输入**可以多于**
  参数：形状链的常量与操作数并列暴露，不匹配任何参数的输入被忽略；同一参数被多个输入命中
  时拒绝绑定。context-ABI 站点最多 14 个参数（rank 0–4，f32/i64），输出为 f32；重复 rank
  或多输出站点必须成功解析名称，否则在申报前丢弃。
- 输出不在调用参数里：kernel 用 `allocate_output` 申请槽位后自行计算形状（例如
  FTDMM 的输出 `[d0, d1, N]` 只有读到输入描述符才知道），Compute 结束时 EP 校验
  所有声明槽位都已分配。
- `kernel_arg_order` 必须为空；输出槽数与融合子图的输出数必须一致（Compile 期核对）。
- 站点由 ANNC 侧的 fusion pattern 产出，本轮只对 ONNX 导入生效（pattern 要求模块带
  ONNX 前端的 `annc.contract_version` 标记）；TF 侧图不产出该站点，融合行为不变。
- 原生池路径的 `num_threads()` 读取真实 `DegreeOfParallelism`（空池为 1）；公开 API
  路径使用硬件并发数限定的建议上界，因为 C API 不提供池尺寸查询。实际执行由 ORT 调度。

- **谁决定融合边界**：ANNC 工具链（`annc-opt --atir-op-fusion`），EP 不做任何 pattern
  匹配。EP 按报告里的成员节点名单整体申报，`sites.json` 的 `outputs[0].tf_name` 是
  回映射 ORT 节点的键（EP 导出时自命名，故可精确反查）。
- **缓存**：`$ANNC_EP_CACHE_DIR/v1/<签名前2位>/<签名>/{sites.json,[libannc_kernels.so],DONE}`，
  零站点结果不生成共享库；非零站点必须同时存在共享库，
  签名 = EP 版本 ‖ 工具链指纹 ‖ 宿主指纹 ‖ 模型签名。含 `Mul` 与 `MatMul`
  的图保留 initializer 内容，避免前置常量仿射折叠复用过期 kernel；其它图的权重
  仍作为运行时实参，签名只保留结构。冷缓存下一次编译（实测 profile1 约 0.6 s、profile4 约 6 s），
  热缓存与不注册 EP 同量级。

## 构建

手工构建（在本目录 `addons/onnxruntime/` 下执行）需要 ORT 1.26.0 的头文件目录与
ONNX protobuf；正常路径是上面的根构建（`--enable-ort-ep`）。

**只 include `onnxruntime_c_api.h`**：它内含 `onnxruntime_ep_c_api.h`，而后者没有自己的
include guard，重复 include 会让其中每个结构体重复定义。

### 原生线程池与 MLAS

EP 构建时**始终**编译原生池路径，没有开关；它需要宿主 ORT 构建树的 `Release/`
归档。MLAS 是 EP 使用的 kernel GEMM 后端，构建 EP 时自动启用（`ANNC_KERNEL_MLAS`
被强制为 `ON`）：构建期需要 ORT 源码的 `mlas.h`，以及宿主构建的
common、cpuinfo、MLAS、KleidiAI 四个归档。EP 的 packed-B 初始化直接调用私有
MLAS 打包接口，因此插件也必须链接 MLAS/KleidiAI，不能依赖宿主导出这些符号。
运行时由 `ANNC_EP_ORT_BUILD_TREE` 定位同一组归档以链接生成的 kernel。
GEMM-ASM 微内核归档是可选依赖；ORT 的 MLAS 路径无需该归档。

`ANNC_EP_PACKED_B=1` 在 session 创建时打包固定 f32 权重；默认关闭，未设置、
空字符串或 `0` 均关闭，非空且非 `0` 开启。关闭仅使用 raw-B，保留融合；
修改后须新建 session。完整使能与验证步骤见
[构建与部署手册](../../tests/ort-ep-build-and-deploy.md#37-packed-b-特性)。BMM 的 pre-affine
折叠与打包可同时开启：编译库通过
`_mlir_ciface___annc_effective_weight_<kernel_name>() -> uint64_t` 暴露内核实际
使用的权重地址；返回 0 表示仍使用原始 initializer。折叠权重读取入口引用与
内核相同的模块常量，不在 Compute 中重复折叠或打包。旧库缺少该入口时 BMM
安全回退 raw-B；缓存指纹已更新，运行时编译会重建产物。

| CMake 选项 | 启用配置 |
|---|---|
| `ANNC_EP_ORT_BUILD_TREE` | 宿主 ORT 构建树，提供 common/cpuinfo 与 packed-B 所需的 MLAS/KleidiAI 归档；构建 EP 时必需 |
| `ANNC_EP_NATIVE_POOL_ORT_VERSION` | 默认 `1.26.0`，须与已验证的私有布局一致 |
| `ANNC_KERNEL_MLAS` | 构建 EP 时自动为 `ON`，无需手动设置；构建期需要 `mlas.h` 与宿主 MLAS/KleidiAI 归档 |
| `ANNC_MLAS_INCLUDE_DIR` | ORT 源码的 `onnxruntime/core/mlas/inc`，包含 `mlas.h` |

运行时 `ANNC_EP_ORT_BUILD_TREE` 必须指向 ORT 构建树（`<build_dir>/Linux/Release`），
`annc` driver 按 `<tree>/` 与 `<tree>/_deps/*-build/` 的布局定位四个归档。四者
任一缺失时直接中止内核链接并报错，不会生成带未定义 `MlasGemmBatch` 符号的
`.so`。

原生池按 `annc_ep_native_pool.h` 中验证的私有布局读取 operator pool。版本字符串相同
不能证明不同构建的 ABI 兼容，归档必须来自宿主同一次构建。宿主版本不匹配时回落公开
API；原生路径的空池表示串行。

原生路径有 cost 时调用 `TryParallelFor`，只有 grain 时固定块调度，两者都没有时使用
公开 API。EP 始终使用 ORT 的线程池：不提供 `ANNC_EP_NATIVE_POOL` 或
`ANNC_EP_SHARDS` 开关，分片数由内部估计与 ORT 池决定。

四个融合算子可通过运行时环境变量逐个开启（默认关闭，设为非 `0` 值开启）：

| 变量 | 作用 |
|---|---|
| `ANNC_EP_FUSION_TENSORDOT_MATMUL=1` | 申报 FusedTensordotMatmul（`tensordot_matmul`）站点 |
| `ANNC_EP_FUSION_LAYER_NORM=1` | 申报 LayerNormalization（`layer_norm`）站点 |
| `ANNC_EP_FUSION_MASKED_MUL=1` | 申报 FusedMaskedMul（`masked_mul`）站点 |
| `ANNC_EP_FUSION_BATCHED_MATMUL_ADD=1` | 申报 FusedBatchedMatmulAdd（`batched_matmul_add`）站点 |

开关由 `annc-onnx-pipeline` 转发给 `annc-opt`，`OpFusion` 在编译期直接跳过对应的
matcher，因此不会生成该融合算子；EP 申报循环保留过滤作为兜底。四个开关进入 EP 缓存
指纹，切换会产生新的缓存条目。注意：开关只关闭该特定融合算子，其他 pattern（例如通用
`matmul`）仍可能融合同一子图中的节点。

FTDMM/BMM 仅在编译启用 MLAS 且 `native_pool()` 非空时调用 `MlasGemmBatch`；否则执行
标量行路径，空池（如 intra-op=1）也不执行串行 MLAS。driver 将上述归档路径写入 kernel
链接命令；EP 与 kernel 必须同源构建。

### 融合覆盖与浮点顺序

BMM 的 rank-3/rank-4 入口均支持把常量 `Mul → Add` 前仿射折叠进 W/bias。
常量按 ONNX 后对齐规则在 batch/K 轴广播，M 轴必须为1；权重发生 batch 广播时保留
原仿射链，不扩张权重。验收须同时核对 site 数、成员节点数和实际 EP 执行事件：
相同的50个 BMM site 可以分别覆盖100或144个原图节点，不能只按 site 数判断等价。
仿射折叠会改变浮点运算顺序，数值验收沿用 `rtol=1e-3, atol=1e-7`，不承诺逐位一致。

BMM 在当前 MLAS 单 K tile 的保守范围 `K<=128` 预填 bias 并使用 beta=1；更大 K
保持 GEMM 完整归约后加 bias，避免 `2^24 + 1 - 2^24` 把 bias 舍掉。
masked_mul 在 AArch64 显式使用 NEON，保持既有输入归约顺序、真实除法以及 NaN/Inf/符号零
处理，余数列仍走标量尾部。ContextCall 在首次成功绑定后发布为只读，动态 shape、输出和
临时分配仍在每次调用内独立创建；同 Session 首调用并发有独立回归。

## 运行

```shell
export ANNC_EP_PIPELINE=/abs/path/bin/annc-onnx-pipeline   # 缺省：本 .so 同目录 → ../bin → PATH
export ANNC_EP_CACHE_DIR=/abs/path/cache                    # 缺省：$XDG_CACHE_HOME/annc-ep
# 会话侧（宿主自己就是我们的进程，不需要 LD_PRELOAD）：
#   env.RegisterExecutionProviderLibrary("ANNC", "<abs>/libannc_ep.so");
#   从 env.GetEpDevices() 里按 EpName() == "AnncExecutionProvider" 过滤
#   session_options.AppendExecutionProvider_V2(env, devices, {});
```

`RegisterExecutionProviderLibrary` 的路径**必须是绝对路径**：相对路径会被拼到别处
（`ORT_LIB_DIR/./libannc_ep.so`）导致加载失败。工具链必须与该 EP 配套安装（
`annc-onnx-pipeline` 及其兄弟工具、`ANNC_KERNEL_LIB_PATH` 指向的归档）。
若目标 clang 依赖系统 GCC runtime，调用进程还应提供对应的 `LIBRARY_PATH`
（例如包含 `crtbeginS.o`、`libstdc++` 和 `libgcc` 的目录）；pipeline 会将该变量
传递给内部编译子进程。

## 最小端到端验证

配置 `ANNC_ENABLE_ONNXRUNTIME_EP=ON` 且使用含 `onnx`、`numpy`、`onnxruntime`
的 Python 环境后，运行
`ctest -R 'AnncEp' --output-on-failure`。
`AnncEpAotFixture` 构建一个单节点 MatMul 模型，通过真实 ORT EP 完成图导出、
site 校验、缓存、融合节点、kernel 加载和数值执行；测试 kernel 的调用标记用于
区分 ANNC 执行与 CPU 回退。fixture 仅代替当前 JIT 专用的 MatMul 代码生成，
不代表真实 `annc-onnx-pipeline` 已能生成 AOT MatMul kernel。

真实 `AnncEpFtdmmAotE2e` 使用 16 节点 ONNX 子图，通过真实 pipeline 生成
`tensordot_matmul` AOT kernel，并验证冷/热 session 与 CPU EP 的数值一致。
架构、缓存配置隔离和失败路径见 [EP 接入与执行设计](../../docs/design/onnx-support/ep-design.md)。

`AnncEpContextFixture` 用同样的方式覆盖 `annc_execution_v2` 路径：`normal` 校验
上下文分配（输出 dims 来自运行时描述符，模型 batch 维动态）、临时缓冲与数值；
`threads` 校验 intra-op=1 时串行、intra-op=4 时内核确实跑在 ORT 工作线程上；
`nested` / `concurrent` 校验嵌套区不扇出与同进程两 session 并存；`error` /
`bad-slot` 校验两类失败都以 `OrtStatus` 冒泡且宿主存活。
`AnncEpFtdmmContextE2e` 是真实闭环：本树的 `annc-onnx-pipeline` 对 profile1 抽出的
16 节点 FTDMM 模型产出 `abi=annc_execution_v2` 站点与本树 kernel 库，EP 经执行
上下文执行并与 CPU EP 对照（需要构建 `annc-onnx-pipeline` 目标）。执行上下文
ABI、参数绑定规则与失败形态见
[EP 接入与执行设计](../../docs/design/onnx-support/ep-design.md)。

调试与测量开关：

| 变量 | 作用 |
|---|---|
| `ANNC_EP_DEBUG=1` | 编译诊断：签名/缓存命中、逐站点申报与丢弃原因、每融合节点的锚点名与 kernel 名、每节点首次操作数形状 |
| `ANNC_EP_TIMING=1` | 进程退出时打印每节点的开销拆分（输入识别 / 输出分配 / kernel） |
| `ANNC_EP_NOOP=1` | `Compute` 只分配输出不做计算——测量辅助，不是运行模式 |
| `ANNC_EP_FUSION_TENSORDOT_MATMUL=1` | 开启 FusedTensordotMatmul 站点申报 |
| `ANNC_EP_FUSION_LAYER_NORM=1` | 开启 LayerNormalization 站点申报 |
| `ANNC_EP_FUSION_MASKED_MUL=1` | 开启 FusedMaskedMul 站点申报 |
| `ANNC_EP_FUSION_BATCHED_MATMUL_ADD=1` | 开启 FusedBatchedMatmulAdd 站点申报 |
| `ANNC_EP_COMPILE=0` | 完全禁用运行时编译（等价于不注册 EP，用于对照） |
| `ANNC_EP_COMPILE_TIMEOUT_S` | 子进程超时，默认 600 |
| `ANNC_EP_KEEP_TEMPS=1` | 保留编译工作目录 |

## 对 ANNC 侧报告的三条硬要求（源自 EP 侧的既有教训）

报告（`sites.json`）必须满足下列条件，否则会静默算错或变慢——这三条都已在 ANNC
侧的 matcher/fusion pass 里实现（`OpFusion.cpp`），此处记录以便新增 pattern 时不回退：

1. **融合集合是完整块，不是最小集合**。`AddNodesToFuse` 不带死代码清理：只提交
   `{flatten Reshape, MatMul, final Reshape}` 时失效的形状链仍留在图上照常执行，实测
   比不融合还慢（0.1595 ms vs 0.0461 ms，base 0.1509 ms）。成员名单必须是闭包给出的整块。
2. **闭包止于边界输入的生产者**。反向生产者闭包若不止于源张量生产者，会把源张量整条
   上游链路一并吞进融合节点——那些节点随即不再执行，形状上看不出来（真模型上实测
   闭包从 16 涨到 79）。
3. **集合内不得有结果逃逸**。ORT 会把被集合外消费的值合法地变成融合子节点的**额外
   输出**，而 mlir_ciface 路径只写 output 0（`ComputeImpl`），下游读到的是未初始化
   内存。ANNC 侧的对应守卫是 `collectEscapingResults`（只允许一个逃逸结果 = 站点
   输出；共享常量/缓冲不算逃逸）；v2 路径的对应守卫是「声明输出数 == 融合子图输出
   数」，在 `CompileImpl` 逐站点核对。**不一致不是只丢该节点**：该节点计入
   `unresolved`，`CompileImpl` 随后整体返回 `ORT_FAIL`，整个 session 创建报错并在
   stderr 打出站点名——`Compile` 阶段没有"只拒一个节点"的手段，取保守的全量失败。

## 一个 EP 侧 API 坑（不是编译期可查的）

`ort_version_supported` 设为 `ORT_API_VERSION` 后，`GetVendorId`、`GetVersion` 以及
   allocator / data-transfer / stream 一组回调必须非空。头文件称其 optional，但 ORT 的
   provider bridge 不判空：`GetVendorId` 在 bridge 构造函数里、`GetVersion` 在
   `CreateEpDevice` 内被调（`plugin_ep/ep_factory_provider_bridge.h:23`、
   `plugin_ep/ep_api.cc:65`）。留空会让 `RegisterExecutionProviderLibrary` 崩在 `0x0`。
