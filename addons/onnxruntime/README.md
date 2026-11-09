# addons/onnxruntime — ANNC 接入 ONNX Runtime

本目录是与 `tensorflow_addons/`（TF 侧接入）对称的 ONNX Runtime 侧接入实现：一个
对 ORT 头文件编译的独立共享库（EP 插件），**默认不参与 ANNC 根构建**：默认
`./build.sh` 不查找 ORT SDK、也不改变既有构建行为。

`context/` 集中 ORT 执行上下文与线程池适配：
`context/annc_ep_execution.*` 和 `context/annc_ep_native_pool.h`。
框架无关 ABI 与服务保留在根目录 `context/`；编译、缓存及站点校验
保留在本目录，测试保留在 `tests/`。

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
> `abi=annc_context_v3`（唯一入参为通用 kernel context，输出经回调分配，线程服务随上下文传入，
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
└─ CreateStateImpl → 创建节点实例；可选 packed-B 打包并绑定实例资源

每次推理
└─ ComputeImpl → 按 site.abi 分发：
     mlir_ciface        → OrtKernelContext → memref → _mlir_ciface_<kernel_name>
     annc_context_v3    → 构造 AnncKernelContext（输入/属性/输出/资源/服务）→
                          _mlir_ciface_<kernel_name>(ctx)

节点实例销毁
└─ ReleaseStateImpl → 释放实例持有的 packed-B 等资源
```

## annc_context_v3：复用通用 Kernel Context

四类融合 builtin（共 14 个 spec）与 TensorFlow builtin 共用根目录 `context/`
的 C 协议，导出签名为 `AnncStatusCode (*)(AnncKernelContext*)`。
kernel 通过 `getInput(slot)`、`getI64Attr(slot)` 和 `allocateOutput<T>(slot, dims)`
操作数字槽位，context 不传递参数名称或运行时 schema。

`context/annc_ep_execution.h/.cc` 负责 ORT 适配：

| 服务 | ORT 实现与约束 |
| --- | --- |
| 输出分配 | `KernelContext_GetOutput`；规范 slot 映射到实际输出索引，检查 dtype/rank、重复分配与溢出 |
| 临时分配 | ORT allocator；支持显式 ZERO，Compute 结束统一释放 |
| 错误 | 保留首错，结合 kernel 返回码与所有输出绑定状态生成 `OrtStatus` |
| 调度 | 同步 `parallel_for` / `shard`，沿用原生 operator pool 与公开 API 路径，嵌套串行 |
| 原生池 | 可选 C 尾回调 `native_thread_pool("onnxruntime")`，供 MLAS 借用，缺失或不可用为 null |

`site.args` / `site.outputs` 按 schema 的规范顺序报告。EP 在初始化/绑定阶段按图名称
解析输入/输出映射；无名称时只允许 rank+dtype 唯一确定，二义映射拒绝。
图名称不参与 kernel 的逐次参数查找。每次 Compute 检查实际 tensor 的入口契约后借用数据，
shape/stride、属性、临时空间和执行状态保持到同步调用返回。
通用适配支持 rank0–8 的 f32/i32/i64 输入与 f32 输出，不再有 14 实参 switch 上限；
各 builtin 仍遵守自己的固定类型/rank 契约。LayerNorm epsilon 保持 rank0 tensor，
MaskedMul 的 8 个静态参数经 `context_i64_attrs` 传递。

BMM 常量前仿射折叠保留原图边界名称，公共 lowering 用 private 常量 view 替换对应输入槽，
共享原 context 的输出、属性、资源和服务。普通单 builtin 入口直接转发原 context。
当前不支持多 builtin 组合或任意边界重排。

融合边界、成员节点与输出集合仍由 ANNC pipeline 决定；EP 不做 pattern 匹配。
所有声明输出都必须成功分配，分配得到的 view 须写回对应的 context 输出槽位；
EP 在调用返回后校验其与宿主分配一致。输出数量须与 ORT 融合子图一致。
`kernel_arg_order` 对 v3 必须为空，普通 `mlir_ciface` 路径继续保留。

缓存改用 `$ANNC_EP_CACHE_DIR/context-v3/<签名前2位>/<签名>/`，
保留既有工具链、宿主、模型、常量与配置签名策略。
旧 `annc_execution_v2` 报告被拒绝，不能只改 ABI 标签复用旧库，需重新生成产物。
2026-10-22 已通过 AArch64/openEuler、ORT 1.26.0 功能验证：CTest 215 项通过，
1 项原本禁用的微基准，包含四类真实 ONNX 闭环。性能与业务模型验收待完成。
完整槽位、所有权、兼容、验证日志与复现命令见
[Kernel Context 设计的 ONNX Runtime 接入章节](../../docs/design/runtime/kernel-context-runtime-design.md#9-onnx-runtime-接入)。

## 构建

手工构建（在本目录 `addons/onnxruntime/` 下执行）需要 ORT 1.26.0 的头文件目录与
ONNX protobuf；正常路径是上面的根构建（`--enable-ort-ep`）。

**只 include `onnxruntime_c_api.h`**：它内含 `onnxruntime_ep_c_api.h`，而后者没有自己的
include guard，重复 include 会让其中每个结构体重复定义。

### 原生线程池与 MLAS

EP 构建时**始终**编译原生池路径，没有开关；它需要宿主 ORT 构建树的 `Release/`
归档。MLAS 是 EP 使用的 kernel GEMM 后端，构建 EP 时自动启用（`ANNC_KERNEL_MLAS`
被强制为 `ON`）：构建期需要 ORT 源码的 `mlas.h`，以及宿主构建的
common、cpuinfo、MLAS、KleidiAI 四个归档。EP 在实例初始化时执行 MLAS 打包；EP 与生成的
kernel 库均链接配套归档，不能依赖宿主导出私有 MLAS 符号。
运行时由 `ANNC_EP_ORT_BUILD_TREE` 定位同一组归档以链接生成的 kernel。
GEMM-ASM 微内核归档是可选依赖；ORT 的 MLAS 路径无需该归档。

`ANNC_EP_PACKED_B=1` 使 FTDMM/BMM 在节点实例初始化（`CreateState`）时打包一次
不可变 f32 B，后续 Compute 只复用；默认关闭，未设置、空字符串或 `0` 均关闭。
EP 在 Compile 阶段按站点输入名称确认 constant initializer 并复制权重，BMM 经同一
生成库的 effective-weight accessor 获取前仿射折叠后的实际权重。fixed shape 本身
不能证明不可变；可覆盖 initializer、动态 B、来源无法确认时使用 raw-B。
打包沿用重构前 EP 的 `BuildPackedBPlan`、`BuildPackedState` 和 `PackedOwner`：
每个 slice 单独分配并调用 `MlasGemmPackB`，实例销毁时逐块释放；仅新增资源引用绑定。

实例持有 packed-B，运行时融合调用元数据（`ContextCall.resources`）保存只读资源引用，
构造 context 时绑定独立的 `resources` / `num_resources`，kernel 按资源类型/版本和完整 shape
校验后读取。通用 Context 无 `packed_b` 专用字段；IR `fusion.metadata` 和 `sites.json`
不保存运行时指针或打包内存。所有调用完成后，`ReleaseState` 析构 `PackedOwner` 并释放资源。
`allocate_temp` 仅对一次调用有效，不能用于保存跨 Compute 的 packed-B。

开关在会话初始化时读取，请在创建 session 前设置，已有实例不随之后的环境变化重新打包。
MLAS 不支持或可选打包分配失败时回到 raw-B，保持融合；普通执行失败仍传播错误。
生成库缓存指纹已更新，部署需配套重编 EP、工具链及 kernel 归档。此次已补充本地主机
MLAS 替身/资源转发回归与目标端到端断言；真实 ORT/MLAS 和性能恢复幅度待目标平台验证。
完整使能与验证步骤见[构建与部署手册](../../tests/ort-ep-build-and-deploy.md#37-packed-b-特性)。

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

原生池按 `context/annc_ep_native_pool.h` 中验证的私有布局读取 operator pool。版本字符串相同
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

FTDMM/BMM 仅在编译启用 MLAS 且 `nativePool("onnxruntime")` 非空时调用 `MlasGemmBatch`；否则执行
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

配置 `ANNC_ENABLE_ONNXRUNTIME_EP=ON`、`BUILD_TESTING=ON` 且使用含 `onnx`、`numpy`、
`onnxruntime` 的 Python 环境，构建后先用 `ctest --test-dir build -N` 核对注册项，
再运行 `ctest --test-dir build -R 'AnncEp' --output-on-failure`。
`AnncEpAotFixture` 构建一个单节点 MatMul 模型，通过真实 ORT EP 完成图导出、
site 校验、缓存、融合节点、kernel 加载和数值执行；测试 kernel 的调用标记用于
区分 ANNC 执行与 CPU 回退。fixture 替代代码生成步骤，仅验证宿主接入，
不能作为真实 `annc-onnx-pipeline` 生成 AOT MatMul kernel 的验证证据。

真实 `AnncEpFtdmmAotE2e` 使用 16 节点 ONNX 子图，通过真实 pipeline 生成
`tensordot_matmul` AOT kernel，并验证冷/热 session 与 CPU EP 的数值一致。
架构、缓存配置隔离和失败路径见 [EP 接入与执行设计](../../docs/design/onnx-support/ep-design.md)。

`AnncEpContextFixture` 用同样的方式覆盖 `annc_context_v3` 路径：`normal` 校验
上下文分配（输出 dims 来自运行时描述符，模型 batch 维动态）、临时缓冲与数值；
`threads` 校验 intra-op=1 时串行、intra-op=4 时内核确实跑在 ORT 工作线程上；
`nested` / `concurrent` 校验嵌套区不扇出与同进程两 session 并存；`error` /
`bad-slot` 校验两类失败都以 `OrtStatus` 冒泡且宿主存活。
`AnncEpFtdmmContextE2e` 是真实闭环：本树的 `annc-onnx-pipeline` 对 profile1 抽出的
16 节点 FTDMM 模型产出 `abi=annc_context_v3` 站点与本树 kernel 库，EP 经执行
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
   输出；共享常量/缓冲不算逃逸）；context v3 路径的对应守卫是「声明输出数 == 融合子图输出
   数」，在 `CompileImpl` 逐站点核对。**不一致不是只丢该节点**：该节点计入
   `unresolved`，`CompileImpl` 随后整体返回 `ORT_FAIL`，整个 session 创建报错并在
   stderr 打出站点名——`Compile` 阶段没有"只拒一个节点"的手段，取保守的全量失败。

## 一个 EP 侧 API 坑（不是编译期可查的）

`ort_version_supported` 设为 `ORT_API_VERSION` 后，`GetVendorId`、`GetVersion` 以及
   allocator / data-transfer / stream 一组回调必须非空。头文件称其 optional，但 ORT 的
   provider bridge 不判空：`GetVendorId` 在 bridge 构造函数里、`GetVersion` 在
   `CreateEpDevice` 内被调（`plugin_ep/ep_factory_provider_bridge.h:23`、
   `plugin_ep/ep_api.cc:65`）。留空会让 `RegisterExecutionProviderLibrary` 崩在 `0x0`。
