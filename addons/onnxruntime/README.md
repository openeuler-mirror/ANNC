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

> 状态：**运行时编译是唯一形态**（2026-09-29 起）。当前 MR2 接入切片只接受
> `pattern=matmul/tensordot_matmul`、`abi=mlir_ciface`、AOT site，并按固定 3D lhs + 2D rhs + 3D output
> 调用；`layer_norm`、`masked_mul`、`batched_matmul_add` 和其他 ABI 必须先完成独立
> kernel/metadata/Compute 契约后才能申报。四模型性能数据不属于本切片验收。

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
└─ CompileImpl → 会话站点表取 kernel 名 → 填 ComputeInfo
└─ ComputeImpl → OrtKernelContext → memref → _mlir_ciface_<kernel_name>
```

- **谁决定融合边界**：ANNC 工具链（`annc-opt --atir-op-fusion`），EP 不做任何 pattern
  匹配。EP 按报告里的成员节点名单整体申报，`sites.json` 的 `outputs[0].tf_name` 是
  回映射 ORT 节点的键（EP 导出时自命名，故可精确反查）。
- **缓存**：`$ANNC_EP_CACHE_DIR/v1/<签名前2位>/<签名>/{sites.json,[libannc_kernels.so],DONE}`，
  零站点结果不生成共享库；非零站点必须同时存在共享库，
  签名 = EP 版本 ‖ 工具链指纹 ‖ 宿主指纹 ‖ 模型结构签名（丢弃权重内容——kernel 把权重
  当运行时实参）。冷缓存下一次编译（实测 profile1 约 0.6 s、profile4 约 6 s），
  热缓存与不注册 EP 同量级。

## 构建

手工构建（在本目录 `addons/onnxruntime/` 下执行）需要 ORT 1.26.0 的头文件目录与
ONNX protobuf；正常路径是上面的根构建（`--enable-ort-ep`）。

**只 include `onnxruntime_c_api.h`**：它内含 `onnxruntime_ep_c_api.h`，而后者没有自己的
include guard，重复 include 会让其中每个结构体重复定义。

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
的 Python 环境后，运行 `ctest -R 'AnncEp(SiteValidation|AotFixture|FtdmmAotE2e)' --output-on-failure`。
`AnncEpAotFixture` 构建一个单节点 MatMul 模型，通过真实 ORT EP 完成图导出、
site 校验、缓存、融合节点、kernel 加载和数值执行；测试 kernel 的调用标记用于
区分 ANNC 执行与 CPU 回退。fixture 仅代替当前 JIT 专用的 MatMul 代码生成，
不代表真实 `annc-onnx-pipeline` 已能生成 AOT MatMul kernel。

真实 `AnncEpFtdmmAotE2e` 使用 16 节点 ONNX 子图，通过真实 pipeline 生成
`tensordot_matmul` AOT kernel，并验证冷/热 session 与 CPU EP 的数值一致。
架构、缓存配置隔离和失败路径见 [特性设计文档](../../docs/Feature/annc-ort-ep-pipeline-adaptation-design.md)。

调试与测量开关：

| 变量 | 作用 |
|---|---|
| `ANNC_EP_DEBUG=1` | 编译诊断：签名/缓存命中、逐站点申报与丢弃原因、每融合节点的锚点名与 kernel 名、每节点首次操作数形状 |
| `ANNC_EP_TIMING=1` | 进程退出时打印每节点的开销拆分（输入识别 / 输出分配 / kernel） |
| `ANNC_EP_NOOP=1` | `Compute` 只分配输出不做计算——测量辅助，不是运行模式 |
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
   输出**，而本 EP 只写 output 0（`ComputeImpl`），下游读到的是未初始化内存。ANNC 侧
   的对应守卫是 `collectEscapingResults`（只允许一个逃逸结果 = 站点输出）。

## 一个 EP 侧 API 坑（不是编译期可查的）

`ort_version_supported` 设为 `ORT_API_VERSION` 后，`GetVendorId`、`GetVersion` 以及
   allocator / data-transfer / stream 一组回调必须非空。头文件称其 optional，但 ORT 的
   provider bridge 不判空：`GetVendorId` 在 bridge 构造函数里、`GetVersion` 在
   `CreateEpDevice` 内被调（`plugin_ep/ep_factory_provider_bridge.h:23`、
   `plugin_ep/ep_api.cc:65`）。留空会让 `RegisterExecutionProviderLibrary` 崩在 `0x0`。
