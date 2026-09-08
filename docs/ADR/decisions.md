# ANNC 架构决策详情

本文件记录 ANNC 已接受的架构决策。新增决策请按编号递增顺序追加到文件末尾，并在 `README.md` 索引中登记。

<a id="adr-001"></a>

## ADR-001：采用统一 IR（ATIR）而非框架专属优化

| 项目 | 内容 |
| --- | --- |
| **上下文** | CPU 场景需支撑多框架客户，若为每个框架单独实现融合+tiling 优化，同一优化策略需重复适配 N 次，且新框架接入时优化能力滞后 |
| **决策** | 引入 ATIR 统一方言作为框架无关的优化层，前端仅负责格式转换，所有优化策略在 ATIR 层实现一次 |
| **后果** | ✅ 优化能力跨框架复用，新增框架仅需开发前端适配器；❌ 前端转换可能丢失框架特有语义，需持续补全 |
| **备选方案** | (a) 为每个框架单独开发优化插件——重复投入大；(b) 基于 TVM Relay 统一 IR——对 CPU tiling/packing 控制力不足 |

<a id="adr-002"></a>

## ADR-002：编译器三层架构（ANNC 框架对接层 / ANNC 工具链前端 / ANNC 工具链后端）

| 项目 | 内容 |
| --- | --- |
| **上下文** | 优化能力碎片化的根因是优化逻辑与框架耦合，框架升级/替换时优化需重做 |
| **决策** | 采用三层架构（ANNC 框架对接层 / ANNC 工具链前端 / ANNC 工具链后端）解耦——框架对接层负责图转换与运行时回接，工具链前端负责图优化与切分决策，工具链后端负责算子接入与代码生成，通过 NodeInfo/Fusion Metadata/Kernel C ABI 三个接口契约连接 |
| **后果** | ✅ CPU 优化能力在工具链后端独立演进，不受前端框架变化影响；✅ 新框架接入仅影响 ANNC 框架对接层；❌ 接口契约需保持稳定，跨层协调成本增加 |
| **备选方案** | 单层紧耦合方案——开发快但演进慢，与解决碎片化问题的目标矛盾 |

<a id="adr-003"></a>

## ADR-003：两层选择算子接入策略

| 项目 | 内容 |
| --- | --- |
| **上下文** | 融合后的子图需要高效执行代码。CPU 微架构优化深度要求高，但不同算子类型的最佳编译策略差异大：GEMM 类算子适合 MLIR 多级 tiling 优化，Embedding Lookup 聚合等非 GEMM 子图走通用 MLIR Lowering，LLM 类算子适合 LLM 辅助自动生成，未来还可能接入 XLA 等外部编译器。同时，已有 vendor kernel（如 KDNN）可直接复用，应优先于自动生成。单一接入路径无法同时满足性能、覆盖度和演进灵活性的要求 |
| **决策** | 算子接入采用两层选择策略：① 编译策略层——由 `Distribute` Pass 根据算子模式选择编译流水线（GEMM MLIR 优化 / 通用 MLIR Lowering / LLM 自动生成 / XLA 接入）；② kernel 实现层——在编译流水线内部，由 `KernelRegistry` 按 op_type+backend+类型约束查询已注册 kernel（vendor 库如 KDNN 优先级高于 aarch64 builtin）；CustomizeCall lowering 无匹配时报错，MLIR 自动生成（Tiling/FastCodegen）为独立 lowering 路径，二者不在同一回退链上。两层选择职责分离，"复用优先、生成为辅" |
| **后果** | ✅ 编译策略与 kernel 实现解耦，各策略独立演进；✅ 已注册 kernel（vendor/builtin）快速复用已有优化，MLIR 路径保证新算子覆盖；✅ 新编译策略可增量接入，不影响已有策略；❌ 两层选择增加分发逻辑和 Kernel 选择复杂度；❌ GEMM/LLM 路径的 Lowering 策略尚未收敛，需持续验证 |
| **备选方案** | (a) 单一 MLIR Lowering 路径——无法覆盖 LLM 自动生成和 vendor kernel 复用；(b) 纯手动接入——覆盖面有限，新算子需逐一开发；(c) 四路径平行分发——手动 kernel 与编译策略不在同一选择层面，强行平行增加理解成本 |

<a id="adr-004"></a>

## ADR-004：fork/exec 进程隔离

| 项目 | 内容 |
| --- | --- |
| **上下文** | MLIR/LLVM 与 TensorFlow 在同一进程内存在符号冲突和 ABI 污染风险 |
| **决策** | `ANNCOptimizer` 通过 `fork/exec` 子进程调用 `annc-tf-pipeline`，实现进程级隔离 |
| **后果** | ✅ 完全隔离 MLIR/LLVM 与 TF 的符号空间；✅ 编译崩溃不影响 TF Serving 主进程；❌ 进程创建开销和临时文件 I/O 开销 |
| **备选方案** | (a) dlopen 隔离——符号隔离不彻底；(b) RPC 远程调用——引入网络依赖和部署复杂度 |

<a id="adr-005"></a>

## ADR-005：运行时无 fallback 决策

| 项目 | 内容 |
| --- | --- |
| **上下文** | 编译阶段失败可回退到 TF 原生执行（§2.3 可靠性价值），但运行时 `.so` 加载或 kernel 执行失败时，ANNCFusedOp 已替换了原始子图节点，无法回退到 TF 原生算子执行 |
| **决策** | 当前运行时不实现 fallback，`.so` 失败直接报错；未来可通过在 GraphDef 重写时保留原始节点为 fallback\_function 属性实现运行时回退 |
| **后果** | ✅ 实现简单，避免运行时 fallback 路径的语义正确性风险；❌ 运行时失败不可恢复，影响推理服务可用性；❌ 与编译时"失败自动回退"的承诺不对称 |
| **备选方案** | (a) 重写时保留原始子图节点——增加 GraphDef 体积和 Session 构建开销；(b) 运行时 fallback 到 TF 原生 Op——需在 ANNCFusedOp 内部重建原始计算图，复杂度高 |

<a id="adr-006"></a>

## ADR-006：tf2atir 使用已解析 TF 图作为 NodeInfo 前置边界

| 项目 | 内容 |
| --- | --- |
| **上下文** | 原 `StandalonePbParser` 同时加载模型、裁剪图、重写 Identity/输出、推导 dtype/shape 和构造 NodeInfo。Tensor 名称字符串混用节点与输出 slot，类型规则分散，且 builder 的 `CustomizeOp` fallback 可能掩盖无法表达的可达 TF op。 |
| **决策** | `annc-tf2atir` 固定为 `TfModelLoader -> TfGraphParser -> TfTensorResolver -> NodeInfoAdapter -> MLIROpBuilder`。`TensorRef{node, output_index}` 是 TF 前端内部唯一 tensor 标识；resolver 以 GraphDef 直接事实为先、集中本地 op 签名和已解析输入为补充，要求每个 tensor 的 dtype 与 rank 在创建 ATIR 前确定。 |
| **后果** | ✅ 图结构、类型/shape 解析和 ATIR 发射职责隔离；✅ `out:1`、多输出和 control edge 不再依赖字符串重写；✅ 可达不支持 op、未知 dtype/rank 均可在前端明确报错；✅ `Const(DT_STRING)` 以既有 `encoding="string"` 和 `DenseStringElementsAttr` 导入；❌ 需维护本地签名规则表；❌ 动态 string 张量及其运行时 ABI 仍未定义。 |
| **备选方案** | (a) 在旧 parser 中继续增加 op 特判——改动小但职责继续耦合；(b) 链接 TensorFlow OpDef/runtime——类型规则完整但引入运行时依赖和版本耦合；(c) 未知 dtype/rank 延后给 ATIR 推导——当前 ATIR 无法表达。 |

<a id="adr-007"></a>

## ADR-007：TensorFlow 无缓存同步 JIT

| 项目 | 内容 |
| --- | --- |
| **上下文** | Grappler 阶段无法为动态 shape 融合子图预先生成唯一的后端 kernel；运行时必须使用实际输入 shape 完成后端特化。 |
| **决策** | `OpFusion` 按 fusion pattern 在 kernel func 上写入 `annc.execution_mode`。pipeline 只将 AOT func 编译进共享库，并保留完整 fusion-only ATIR；converter 按 fusion mode 将 AOT 节点连接到共享库、JIT 节点连接到 ATIR 模板。`ANNCFusedOp::Compute` 对 JIT 节点提取实际 shape，同步调用 `annc-asm` 和 `annc`，加载生成的共享库并执行 kernel。 |
| **后果** | 动态 shape 的编译决策延后到运行时且边界清晰；首次调用包含编译开销，每次调用都可能重新生成共享库，缓存和并发控制留待后续独立提交。 |
| **备选方案** | 在 Grappler 阶段按样例 shape 预编译——无法覆盖运行时动态 shape；异步 JIT——需要额外的请求排队和失败语义。 |

<a id="adr-008"></a>

## ADR-008：TensorFlow 进程内 JIT 编译缓存

| 项目 | 内容 |
| --- | --- |
| **上下文** | ADR-007 的同步 JIT 能将请求实际 shape 交给后端，但每次调用都重新执行 `annc-asm`、`annc` 和 `dlopen`。直接按 `kernel_name` 缓存会阻止语义相同但来源名称不同的 fusion func 复用。工具链、GEMM 配置和 ABI 是进程级固定上下文，不需要在每次推理时重新探测。 |
| **决策** | 在 `ANNCFusedOp` 进程内增加有界编译缓存。key 由 cache schema version、与名字无关的 canonical ATIR template fingerprint 和按 kernel 参数顺序排列的实际 shape 构成（`annc.intra_thread_count` 曾纳入 template fingerprint，后按 ADR-009 移出）。`kernel_name` 和 MLIR func 名称不进入 key，只在 miss 时选择 func 和解析首次生成的符号。相同 key 使用 single-flight，同步等待同一编译结果；不同 key 可并行编译。成功 entry 以 `shared_ptr` 管理 `dlopen` handle 和工作目录并按 LRU 淘汰，默认上限 64，`ANNC_JIT_CACHE_MAX_ENTRIES=0` 可关闭缓存。失败结果从表中移除，后续请求重新编译。运行时 shape JSON 只携带参数索引和实际 shape，dtype 由 ATIR 模板和 TensorFlow Op dtype contract 共同约束。 |
| **后果** | 相同语义和 shape 在进程内只编译一次；工具链、GEMM 配置和 ABI 变化属于不支持的进程级上下文切换，需要重启进程；正在执行的产物在淘汰后仍存活到最后一个引用释放。首次请求仍承担同步编译延迟，缓存仅限当前进程。 |
| **备选方案** | (a) 每个 Op 仅缓存最后一个 shape——无法跨节点复用且 shape 切换会重复编译；(b) 使用 `kernel_name` 作为 key——把来源身份误当成代码生成语义；(c) 跨进程磁盘缓存——需要额外的原子发布、完整版本校验和不可信 `.so` 安全边界，本阶段不采用。 |

<a id="adr-009"></a>

## ADR-009：GEMM intra 线程数改为运行时 JIT 特化维度

| 项目 | 内容 |
| --- | --- |
| **上下文** | GEMM lowering 的线程分片计划按 intra 线程数特化。原设计由 Grappler 阶段解析线程数（session config → env → MaxParallelism）经 `--intra_thread_count` 写入 `annc.intra_thread_count` module 属性，参与 template fingerprint 并被 SelectGemmStrategy 消费。但 TF 的 `GrapplerItem::optimization_options().intra_op_parallelism_threads` 会把未配置值填成 `port::MaxParallelism()`，与显式配置值不可区分，导致 `taskset` 限核 + `TF_NUM_INTRAOP_THREADS=1` 时生成与实际 pool 不匹配的分片计划；而 kernel 执行时注入的正是 TF intra-op pool，精确值在运行时每次 Compute 都可零成本获得。 |
| **决策** | 线程数的唯一信息源改为 `ANNCFusedOp` 运行时 JIT 编译路径：编译（miss）时读取 `ctx->device()->tensorflow_cpu_worker_threads()->workers->NumThreads()`（无 pool 时串行默认 1），经 `--annc-aarch64-gemm-pipeline=intra-thread-count=N` 传给 annc-asm。移除 `annc.intra_thread_count` module 属性、`annc-tf2atir`/`annc-tf-pipeline` 的 `--intra_thread_count` CLI、ANNCOptimizer 的线程数解析，以及 template fingerprint 中的线程数字段（schema v2→v3）。TF intra-op pool 在 session 创建时初始化且进程内恒定，与 JIT cache 生命周期一致，因此线程数不进入 cache key。AOT 分支不再运行 AArch64 GEMM pipeline（GEMM 融合 kernel 均为 jit 模式，AOT kernel 无 GEMM anchor，属历史污染，一并清理）。 |
| **后果** | ✅ GEMM 计划线程数与实际执行 pool 恒一致，消除 Grappler 信息歧义；✅ fingerprint 不再被编译期猜测值污染，cache key 只含真实语义维度；✅ 线程池恒定性使 key 不膨胀；❌ 同进程多 session 且 intra 配置不同的场景下 cache 会串用首个编译产物（当前部署为单 session，不做防御）；❌ fingerprint schema 变更导致存量部署升级后首次 miss 重编一次。 |
| **备选方案** | (a) 保留 Grappler 解析 + 修补启发式（configured==MaxParallelism 视为未配置）——无法区分显式设置恰好等于 MaxParallelism 的情况，且解析发生在信息已被污染的层；(b) 线程数进 cache key——pool 进程内恒定，只会制造冗余 key；(c) 运行时实测后回写 module 属性——Grappler 已结束，属性无人再读。 |

<a id="adr-010"></a>

## ADR-010：TF 控制流镜像转换（mirror，不做语义重建）

| 项目 | 内容 |
| --- | --- |
| **上下文** | tf2atir 前端此前用 `ANNCStructuredSwitch`（菱形重构）把 TF Switch/Merge 子图重建成单个结构化 op。真实服务图（dead Switch 输出、fan-out 分支、disjoint 分支、嵌套/多条边）大量无法满足菱形重构的前提，导致转换失败或静默丢结构；且 SwitchCase（region 版 switch）与 TF 静态图拓扑一一对应的诉求不符。 |
| **决策** | 前端对 11 个 TF 控制流 op（Switch/RefSwitch、Merge/RefMerge、Enter/RefEnter、Exit/RefExit、NextIteration/RefNextIteration、LoopCond）一一镜像为显式 ATIR op：`atir.switch`、`atir.merge`、`atir.enter`、`atir.exit`、`atir.next_iteration`、`atir.loop_cond`，不做语义重建、不降级 opaque；`ANNCStructuredSwitch` 与 region 版 SwitchCaseOp 移除。具体约束：(1) 拓扑排序无条件剥离 NextIteration→Merge 数据回边——NextIteration 在 TF 中仅出现于循环构造且回边必汇入 Merge，剥离是无条件正确的；builder 只在 producer 已构建时接线，其余通过 `tf.input.N` metadata 保留来源。(2) 控制依赖（`^input`）纳入可达性与拓扑排序 indegree——控制边影响执行顺序，必须参与；TF 静态图本身要求 DAG，控制环从"被忽略"改为显式失败属于纠偏。(3) 缺 `_output_shapes` 时不再失败：mirror op、Identity 桥接、未知 op 的缺失 slot 从输入合成 descriptor（Switch/Merge 固定双输出，Merge `value_index` 固定标量 i32），以未知 rank 表达——结构保真优先于早期失败，下游 fusion 以 atir.opaque 语义吸收。(4) mirror op 的定位是**保拓扑、不执行**：Conversion/、Target/aarch64/、Interpret/ 不为它们提供 lowering/解释；完整 AOT lowering 只作用于 prune 后的纯计算 func，mirror 结构供 OpFusion 吸收与 converter（GraphDef 重写）恢复。 |
| **后果** | ✅ 任意复杂控制流拓扑（嵌套 Switch、fan-out/fan-in、disjoint 分支、Ref 全家族、循环回边）可无损转换并通过 FileCheck/pytest 验证；✅ RankInference 跨镜像 op 等价传播保持 shape 推导能力；❌ 图中 mirror op 若未被 fusion 吸收并走到 AOT lowering 会失败（预期行为：该子图本就不应 AOT 编译）；❌ 控制环图与仅 control 可达的子图行为比旧版严格。 |
| **备选方案** | (a) 继续菱形重构——前提过强，真实图大量不满足；(b) 控制流降级为 atir.opaque——丢失控制流语义，converter 无法恢复图结构；(c) region 版 SwitchCase——与 TF 静态拓扑不对应，需要额外的 region 语义。 |
