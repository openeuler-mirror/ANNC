# 运行时接入

| 文档控制项 | 内容 |
| ----- | ----- |
| 文档编号 | ANNC-ARCH-002 |
| 版本与日期 | 1.0，2026-10-20 |
| 文档状态 | 架构说明。描述 TensorFlow 与 ONNX Runtime 两条运行时接入路径的产物契约与执行方式 |
| 设计基线 | `tensorflow_addons/`、`addons/onnxruntime/`（源码基线 `11bef0d0`，分支 `upstream/dev`） |
| 责任人 | 编译器开发团队 |
| 上级文档 | [ANNC 架构设计文档](overview.md) |

回答「ANNC 的编译产物如何被推理框架在运行时消费」。ANNC 对 TensorFlow 通过 Grappler 插件 + 自定义 Op 接入，对 ONNX Runtime 通过 EP 插件接入；两者都以编译期产物（`.so` 或 ATIR module）为输入。

## TensorFlow 接入

`tensorflow_addons/` 提供 Grappler 插件和 `ANNCFused` 自定义 TensorFlow Op。

| 文件 | 说明 |
|------|------|
| `tensorflow_addons/annc_optimizer.cc` | Grappler 优化器插件，自动调用 `annc-tf-pipeline` 重写计算图 |
| `tensorflow_addons/annc_fused_op.cc` | `ANNCFused` Op，运行时 `dlopen` 加载 `.so` 并调用 `_mlir_ciface_` 接口 |
| `tensorflow_addons/annc_fused_op_register.cc` | 注册 `ANNCFused` Op |
| `tensorflow_addons/annc_optimizer_register.cc` | 注册 Grappler 优化器 |

### AOT 与 JIT

Serving 场景下，`ANNCOptimizer` 调用 `annc-tf-pipeline` 后会保留 pipeline work_dir 中的产物，尤其是 AOT 共享库 `annc_generated_kernel.so` 与 fusion-only ATIR。重写后的 GraphDef 按 fusion 的 `execution_mode` 写不同 attr：**AOT 节点**写 `shared_lib_path`（指向该 `.so`）且 `atir_module_path` 为空，运行时 `dlopen` 加载；**JIT 节点**（GEMM 家族）写 `shared_lib_path=""` 与 `atir_module_path`（fusion-only ATIR 的绝对路径），运行时按实际 shape 现场编译，因此 AOT 节点需保留 `.so`，JIT 节点需保留 ATIR module；纯 JIT 模型可能不产生 `.so`。`annc` driver 生成 object/LLVM IR 的临时目录使用微秒时间、进程号和重试序号组成唯一目录，支持多个 Serving 实例或多个 Grappler 优化任务并发编译，避免不同进程删除彼此的 `step*.ll` 中间文件。

GEMM fusion 的 GraphDef 节点写入 `atir_module_path` 和与名字无关的 `template_fingerprint`，运行时根据实际输入和输出 shape 选择并特化当前 fusion func，再同步执行 `annc-asm` 和 `annc`。非 GEMM fusion 节点写入 AOT 共享库路径。进程内缓存按 `OpFusion` 生成的 `template_fingerprint`（对规范化后的 kernel func 计算，不含 module 级属性）和实际参数 shape 复用 `.so`；常量折叠构建的身份另含私有 packing ABI 版本、执行 ABI 和不可变参数槽，不包含权重内容。工具链、GEMM 配置和线程预算固定于进程上下文，`kernel_name` 不进入 key。相同 key 的并发 miss 只编译一次，不同 key 可并行编译。Ready entry 使用有界 LRU，默认保留 64 个（`ANNC_JIT_CACHE_MAX_ENTRIES`）；编译失败不缓存，后续请求可重试。

启用 `ANNC_ENABLE_CONSTANT_FOLDING` 构建后，JIT 固定采用代码与权重缓存分离：共享库只含计算代码、packing 布局和准备入口，各 `ANNCFusedOp` 从首次输入保留权重 Tensor，并独立持有 packed buffer。无运行时模式选择、权重哈希或快照文件。候选 RHS 必须在 OpKernel 生命周期内不可变；换模型创建新实例。fixed 形状角色不证明内容不可变；每个节点首次初始化复用旧静态 prepack 的 checkpoint 名称、F32 类型和字节数检查，未通过的输入使用普通 runtime-packed/direct。准入在全局代码缓存查找前完成，连续或交替 shape 命中均不重复读取 checkpoint。checkpoint 不证明运行期不可变，已准入权重仍不支持原地更新。跨 batch 复用要求输入槽、K/N/KC/NC/ISA/panel 等完整布局一致，切块改变则重新准备对应布局。连续同 shape 直接复用已有 executable/binding，不查权重缓存、不检查初始化 once_flag；shape 切换才查节点绑定，新布局才分配和打包。编译期开关关闭时保持普通 runtime-packed/direct 路径。后端通过 `jit-inputs`（不可变 kernel 参数索引的 JSON 数组，例如 `[1]`）与 `packed-c` 生成准备代码，沿用 `annc --packed-rhs-c` 链接入口。详见[分离缓存实现契约](../design/jit-constant-folding/cache-separation.md)。

原始输入名通过既有 annc-fusion-metadata 从 ATIR 提取，不新增 GraphDef 属性；
工具或元数据不可用时使用普通 packing。新 Op 首次命中共享代码也需初始化，
元数据提取和 checkpoint 读取不进入后续缓存命中路径。

### 图重写与形状推导

多 fusion 的 GraphDef 重写会为每个 fusion 生成一个独立的 `ANNCFused` 节点。`fusion_metadata.json` 中如果某个 fusion 带有 `dynamic_dims`，`annc-converter` 写入 `ANNCFused.output_shapes` 时会把这些维度保留为 `?`，由 `ANNCFused` 在运行时根据动态输入的实际维度推导输出形状，避免 Serving 请求 batch 与编译样例 batch 不一致时按固定维度分配输出。

MatMul fusion metadata 保持 `ANNCFused` 的运行时输入约定：fixed shape 输入在前，dynamic shape 输入在后。`OpFusion` 会根据 MatMul 的 lhs/rhs 哪个输入承载输出 batch 维来生成 `kernel_arg_order`，保证 GraphDef 输入顺序满足运行时形状推导，同时 `_mlir_ciface_` 调用仍按 kernel 函数签名接收 lhs、rhs、output。

### 日志与输出初始化

`ANNCFused` 运行时默认只保留错误日志，避免 Serving 并发请求时 INFO 调试日志影响吞吐和日志可读性。`dlopen`/`dlsym` 以及兼容 `mlir_ciface` 路径的 memref 构造、输出分配、参数顺序和 kernel 调用失败会输出 `LOG(ERROR)`；上下文 ABI 路径的错误通过 status/执行上下文返回，不保证写日志。若另有 Serving 侧的 addon 副本（`tensorflow_serving_addons`，不在本仓库），需与 `tensorflow_addons` 保持对应修改一致。

`ANNCFused` 通过 `zero_initialize_outputs` 控制输出 Tensor 是否在调用 kernel 前清零。该 attr 默认 `true`，保持旧 GraphDef 和非全量写输出 kernel 的保守行为；运行时对 `fusion_pattern=dnn_embedding_hash_bucket` 自动关闭清零，因为生成的 embedding kernel 会完整覆盖 `[batch, embedding_dim]` 输出，跳过清零可消除 `avg_output_init` 带来的主要包装层开销。该优化不要求 GraphDef 写入新 attr，避免不同 Serving OpDef 版本下 unknown attr 被忽略。

### 运行时性能打点

`ANNCFused` 运行时性能打点默认关闭，排查融合 Op 延迟时可通过 `ANNC_FUSED_PROFILE` 与 `ANNC_FUSED_PROFILE_INTERVAL` 开启（见[环境变量参考](../guides/environment-variables.md)）。

```bash
ANNC_FUSED_PROFILE=1 ANNC_FUSED_PROFILE_INTERVAL=100000 ...
```

开启后每隔 `ANNC_FUSED_PROFILE_INTERVAL` 次 `ANNCFused` 调用按 profile key 汇总输出平均耗时；未设置时默认每 100000 次输出一次，避免高 QPS 测试时 profile 日志本身干扰压测。key 规则如下：

- MatMul 类融合保持兼容格式：`kernel_name_MxKxN`，如果节点带有 `fusion_pattern`，追加 `pattern=<fusion_pattern>`。
- `dnn_embedding_hash_bucket` 使用 embedding 专用格式：`kernel_name_pattern=dnn_embedding_hash_bucket_weight=<weight_shape>_input=<runtime_input_shape>_out=<metadata_output_shape>`，避免把 embedding weight 和 string/id 输入误解释为 MatMul 的 lhs/rhs。
- 其他融合算子使用通用格式：`kernel_name_pattern=<fusion_pattern>_inputs=<runtime_input_shapes>_out=<metadata_output_shapes>`。

单次调用的耗时采样保存在 `Compute()` 局部变量中，只有汇总统计写入全局表时加锁，避免并发请求时多个线程改写同一个 `OpKernel` 成员导致统计互相覆盖。主要日志字段含义如下（`key`、`count` 为汇总键与调用次数，除标注为次数的回调计数字段外，其余 `avg_*` 均以微秒计）：

| 字段 | 含义 |
|------|------|
| `avg_load_library` | AOT 模式下首次加载或 `.so` 路径变化时 `dlopen`/`dlsym` 的平均耗时，缓存命中时为 0；JIT 命中时为 0，JIT miss/wait 时计入从 shape 构建到 `GetOrCompile`（含编译）的耗时，与 `avg_jit_compile` 部分重叠 |
| `avg_jit_arg_shapes` | JIT 运行时构建参数形状/签名的平均耗时（均摊到总调用数） |
| `avg_jit_cache_key` | JIT 构建 cache key（SHA256）的平均耗时（均摊到总调用数） |
| `avg_jit_cache_lookup` | JIT `GetOrCompile` 每次调用的平均耗时（hit/miss/wait 混合，均摊到总调用数） |
| `avg_jit_hit_lookup` | 命中（含 `kHit` 与 shape fastpath 命中）时查表的平均耗时，按命中次数（`jit_hits`）取平均，衡量稳态命中路径 |
| `avg_jit_compile` | JIT cache miss 时单次编译的平均耗时，按编译次数（`jit_compiles`）取平均（`fork/exec annc-asm` → `annc` → `dlopen`/`dlsym`），衡量首次编译/冷启动成本 |
| `steady_calls` / `avg_steady_total` / `avg_steady_kernel` / `avg_steady_overhead` | 稳态热路径统计：仅统计 JIT cache 命中（或 AOT 已加载）的调用，不含首次编译与 wait 等冷启动成本；`avg_steady_overhead = avg_steady_total - avg_steady_kernel`，衡量稳态接入层开销 |
| `jit_hits` / `jit_compiles` / `jit_waits` | JIT cache 命中/编译/等待并发编译的调用计数 |
| `avg_threadpool_setup` | 获取 TensorFlow CPU threadpool 并安装 ANNC TLS threadpool 的平均耗时 |
| `avg_threadpool_restore` | 调用结束后恢复线程池上下文的平均耗时 |
| `avg_backend_dispatch` | 后端分派（如 `openblas` 判断）的平均耗时 |
| `avg_input_memref` | 输入 Tensor 构造成 MLIR memref descriptor 的平均耗时 |
| `avg_output_alloc` | TensorFlow 输出 Tensor 分配的平均耗时 |
| `avg_output_init` | 输出 Tensor 清零初始化的平均耗时 |
| `avg_output_memref` | 输出 Tensor 构造成 MLIR memref descriptor 的平均耗时 |
| `avg_arg_order` | 构造 `_mlir_ciface_` 实参顺序的平均耗时 |
| `avg_execution_setup` | 上下文 ABI 路径下执行上下文准备的平均耗时 |
| `avg_execution_post` | 执行上下文收尾（执行后处理）的平均耗时 |
| `avg_kernel` | `_mlir_ciface_<kernel_name>` 真正执行生成 kernel 的平均耗时（含执行回调） |
| `avg_execution_callbacks` | kernel 执行期间回调（输出/临时缓冲分配）的平均耗时 |
| `avg_tf_output_alloc` / `avg_tf_temp_alloc` | 回调中 TensorFlow 输出缓冲与临时缓冲分配的平均耗时 |
| `avg_callback_framework` | 回调中框架侧开销（`avg_execution_callbacks` 减去上述两项分配）的平均耗时 |
| `avg_kernel_excluding_callbacks` | `avg_kernel` 扣除执行回调后的纯 kernel 平均耗时 |
| `avg_output_callbacks` / `avg_temp_callbacks` | 每次调用触发的输出缓冲与临时缓冲回调次数 |
| `avg_total` | `ANNCFusedOp::Compute` 端到端平均耗时 |
| `avg_overhead` | `avg_total - avg_kernel`，用于判断 TensorFlow Op 包装层开销 |

## ONNX Runtime 接入

ONNX Runtime 侧通过 EP 插件（`addons/onnxruntime/`）在 `GetCapability` 内导出整图、调用 [`annc-onnx-pipeline`](../guides/cli-workflows.md) 编译，并按站点报告 `sites.json` 精确申报融合区域。接入契约、缓存与降级设计见 [ONNX Runtime EP 接入设计](../design/onnx-support/ep-design.md)，编译侧见 [ONNX 前端设计](../design/onnx-support/frontend-design.md)。
