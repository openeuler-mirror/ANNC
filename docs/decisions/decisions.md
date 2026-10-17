# ANNC 架构决策详情

本文件记录 ANNC 已接受及待评审的架构决策；待评审条目标为 proposed，不表示已经采纳。新增决策请按编号递增顺序追加到文件末尾，并在 `README.md` 索引中登记。

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

状态：已由 ADR-011 替代。以下保留原始决策背景。

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

<a id="adr-011"></a>

## ADR-011：ONNX 前端接入与"不可表达即降级"契约

| 项目 | 内容 |
| --- | --- |
| **上下文** | 需要把 ONNX 模型接入既有 ATIR 优化与后端，而不是另起一套框架专属优化。可复用的部分只有 ATIR 与共享 builder 引擎（`MLIRBuilder`）：TF 与 ONNX 对**同名算子**语义不同（ONNX `Sum` 是变参逐元素加、TF `Sum` 是归约；ONNX `Gather` 的 axis 是 attribute、TF `GatherV2` 是第三个 operand），直接共用一张 op 规格表必然静默错译。此外 ONNX 的算子 schema 随 opset 漂移（`Squeeze`/`ReduceSum` 的 axes 在 opset 13 从 attribute 变 operand、`Softmax` 的默认 axis 从 1 变 -1），而 ATIR 当前的数值语义是 float32 优先（`Interpret/` 的元素级实现统一走 `getFloatValues`，`Target/aarch64` 无整型算术路径），两者之间存在无法靠 op 名对齐的表达差距。 |
| **决策** | (1) **位置**：新框架前端放 `frontends/<framework>/`，由 `frontends/CMakeLists.txt` 按 `ANNC_ENABLE_<FRAMEWORK>` 注册，不进 `annc/tools/`（避免核心目录膨胀，新增框架不改其它框架的构建路径）。(2) **复用接缝**：前端继承 `MLIRBuilder`，只替换两处虚接缝——`lookupSpec`（ONNX 专表，与 TF 表分离）与 `shouldLowerToOpaque`（逐节点降级判定）；节点校验、op 构造、属性映射、metadata、诊断全部继承。(3) **opset**：只支持 `ai.onnx` opset 13，其它版本默认整体拒绝（表按一个 opset 建立，换版本会静默错译）；`--relaxed-opset` 时按算子敏感表逐节点降级并以 `OP_SET_MISMATCH` 记录。(4) **降级契约**：任何无法忠实表达的节点（无表行、变参 arity 超出表行、不支持的属性/元素类型、rank 未知、opset 漂移）一律降级为 `atir.opaque`，并写入机器可读 reason；配套 provenance 逐节点审计、`nodes_accounted`/`opaque_nodes_match` 守恒校验、`--strict` 与退出码 0/2/1。前端内部状态（降级标记、归一化后的 Slice 窗口）以 `NodeInfo::attrs` 中前导下划线的私有属性传递，构建器保证其**不进入** emitted metadata。(5) **边界**：不做控制流（含 GRAPH 属性的模型整体拒绝而非部分导入，否则守恒账目会出现缺口）；不支持 float8/int4 等 ATIR 无法承载的元素类型（普通节点降级为 `UNSUPPORTED_DTYPE`，initializer 因 payload 无法解码而响亮失败）。(6) **dtype 门禁**：ATIR 数值路径为 float32 优先，因此对 `Div/Pow/Sqrt/Reciprocal/Abs/Min/Max/Mod/Equal/Greater/Less` 的整型输入逐节点降级为 `UNSUPPORTED_DTYPE`；`Add/Sub/Mul/Tile` 等索引与形状算术**暂不设门禁**（同受 float32 约束，但只在绝对值大于 2^24 时失真），作为已知缺口记录，待 ATIR 补齐整型内核后统一放开。 |
| **后果** | ✅ ONNX 前端复用 ATIR 优化与 AArch64 后端，新增框架的成本收敛为"一张表 + 一个适配器 + 一组 transformer"；✅ 没有静默漏节点：每个模型节点、initializer 与图输出都有 provenance 行，覆盖率报告可审计，退出码区分"全映射/有降级/硬错误"；✅ 同名算子语义分歧（`Sum`、`Gather`）被表分离结构性隔离。❌ op 覆盖是目标融合子图所需的**子集**，子图之外的模型会大量落到 `atir.opaque` 或被整体拒绝；❌ 保守的 dtype 门禁牺牲整型模型的映射率（整型 `Min`/`Max`/`Div` 等可直接降级）；❌ `Add/Sub/Mul` 的整型路径仍是潜在静默失真点，需要 ADR 之外的工作（ATIR 整型内核）才能真正闭合；❌ opset 13 之外的模型需要重跑或显式放宽。 |
| **备选方案** | (a) 与 TF 共用一张 op 规格表——同名不同义（`Sum`/`Gather`）会静默错译，正是拆表的理由；(b) 无表行的节点直接报错——与"逐节点降级、绝不整体拒绝"的架构目标冲突，且单点缺失会放大为全模型不可导入；(c) 把 ONNX 前端放进 `annc/tools/`——与 §8「核心模块膨胀控制」冲突，且会把框架差异渗进 TF 前端；(d) 支持多 opset 自适应——需要按 opset 分版本建表，属未来工作。 |

<a id="adr-012"></a>

## ADR-012：ORT EP 由 pipeline 报告驱动申报与运行时编译缓存

| 项目 | 内容 |
| --- | --- |
| **上下文** | MR2 提供 ONNX→ATIR 前端，但 ORT 的图分区需要确定融合成员、参数 ABI 和可加载 kernel。EP 若重复实现 matcher，或先申报再尝试编译，容易与编译器的融合边界不一致，并把可回退的错误变成 session 失败。 |
| **决策** | 由 EP 导出自包含 ONNX，由 `annc-onnx-pipeline` 生成 `sites.json` 和 AOT 库。EP 在 `GetCapability` 内验证报告结构、当前 ABI、成员映射和 kernel 符号后才申报。磁盘缓存按模型结构、工具链和影响编译的配置隔离，同签名 `flock` 单飞，工作目录 rename 后写入 `DONE`。结构损坏与编译失败不发布成功缓存；零站点允许成功缓存且无需共享库。 |
| **后果** | matcher 只有工具链一份实现；编译和符号失败可通过不申报保留 CPU EP。首次 session 承担冷编译成本，后续复用缓存。当前真实执行验证覆盖 AOT 四个上下文站点（`tensordot_matmul`、`masked_mul`、`layer_norm`、`batched_matmul_add`，均随 `!249` 合入）；普通 MatMul JIT、其他 pattern、控制流子图和 ORT 执行上下文扩展仍需独立实现。缺符号会放弃 site，不自动重编译已发布缓存。 |
| **备选方案** | EP 自行匹配融合边界会重复 compiler 逻辑；在 ORT `Compile` 才编译会失去不申报回退的时机；完全离线编译需要额外的模型与缓存分发协议。 |
| **源码依据** | ANNC `b3513b7d`：`GetCapabilityRuntimeCompileImpl`、`CompileDriver::Compile`、`ReadReport`、`ValidateSite`；详细设计见 [EP 接入与执行设计](../design/onnx-support/ep-design.md)。 |

<a id="adr-013"></a>

## ADR-013：ONNX EP 执行上下文——线程池入上下文、TF/ONNX 共用 ABI v3

| 项目 | 内容 |
| --- | --- |
| **上下文** | ONNX Runtime 侧的融合回接需要一套宿主执行状态：输出必须由 kernel 经 `KernelContext_GetOutput` 在运行中分配（输出形状常要读到输入才知道），线程池只能经 `KernelContext_ParallelFor` 逐索引回调获得，ORT 进程里没有 ANNC 的 TLS 约定。既有 `annc_execution_v2` 上下文（TF 域 pattern 已在用）没有线程池字段，TF 靠 TLS 取池；ONNX 无 TLS 可取，线程池实际拿不到——这是 dev 分支实现暴露的问题，也是本决策要解决的输入。 |
| **决策** | (1) **ABI 扩展为 v3**：`AnncExecutionContext` 尾部新增不透明 `AnncThreadPoolHandle* thread_pool`（NULL = 未提供）；`validateContext` 同时接受 v3（`sizeof` + 版本 3）与 v2（`ANNC_EXECUTION_V2_CONTEXT_SIZE` + 版本 2）并拒绝其余错配，保证旧宿主的小结构不会被越界读。(2) **取池语义**：`threadPool(context)` 只取上下文携带的池；`resolveThreadPool(context)` 上下文池优先、回退 TLS。TF 与 ONNX **共用同一个上下文抽象**（曾评估 ONNX 私有结构体，因尺寸判别/helper/回归面成本被放弃）：TF 构造的上下文池为空 → 回退 TLS，TF 执行流与 kernel 取池方式零改动；ONNX 填池 → kernel 从上下文取。(3) **ONNX 侧执行状态**：EP 内实现 allocate_output / allocate_temp / report_error 三回调与 `OrtThreadPool`（逐索引回调转 ANNC 区间；`grain_size`→批数；`KernelContext_ParallelFor` 失败则释放状态并在本线程内联执行整段；thread_local 深度提供 `in_parallel_region` 供嵌套判定）。(4) **站点按 ABI 分发**：EP 按站点报告的 `abi` 字段选择调用路径（`mlir_ciface` 保持 017 契约；`annc_execution_v2` 走上下文路径），上下文与具体融合算子无关。(5) **契约申报前校验**：上下文站点无 `kernel_arg_order`、参数 1–14 个、rank 0–4、输入 `f32`/`i64`、输出 `f32` 且 ≥1 个并声明 rank/type；重复 rank 或多输出时要求站点名称全部可解析，否则在申报前丢弃；参数解析优先按名称置换（申报期回译原图名，`Compile` 建立参数与输出槽置换），无名称时按 **rank+元素类型** 匹配（不匹配的输入忽略——融合节点的形状链常量与操作数并列暴露；两个输入命中同一参数则首次 Compute fail-closed）；契约与融合子图不符时该节点计入 unresolved、整体 Compile 失败（保守的全量失败，Compile 阶段没有单节点拒绝手段）。 |
| **后果** | ✅ TF 语义与执行流零改动与 ONNX 拿到真实 ORT 池并存，一个 ABI 两种宿主流；✅ kernel 侧取池统一为 `resolveThreadPool`，TLS 成为兼容兜底而非唯一通道；✅ 失败模式收敛为 fail-closed 且可见（契约不符整 session 失败并打站点名、丢弃站点有日志、参数二义拒绝）；❌ EP 与 kernel 必须同源构建——`thread_pool` 句柄背后的 C++ 虚表不在尺寸/版本检查覆盖内；❌ 契约不符的代价是整 session 创建失败而非单节点降级；❌ 无名称站点的参数绑定是 rank+dtype 启发式（命名站点已按名称置换），未来若出现"与参数同 rank 同类型的额外输入"会 fail-closed 拒绝（安全但保守），届时需在契约层显式扩展。后续 `!249` 在适配器内增加了受版本校验的原生 operator pool 路径与默认关闭的 MLAS 路径，私有布局须随宿主版本重新验证。 |
| **备选方案** | (a) ONNX 私有扩展上下文结构体——kernel/EP 各自维护尺寸判别与 helper，回归面与维护成本高于收益，退回共用 ABI；(b) kernel 一律走 TLS、由 EP 用 TLS 注入 ORT 池——ORT 无此约定，且把两条宿主的池语义混为一谈；(c) 站点级单节点拒绝——ORT 的 Compile 契约只支持整批失败，做不到；(d) 参数按报告 `role`/操作数索引精确绑定——多一条上游字段信任链，当前 rank+dtype+二义拒绝已 fail-closed，留作后续演进。 |
| **源码依据** | 本任务分支 `feature/annc-ort-ep-execution-context`：`0fc1b1c2` ABI v3、`6fa260d2` EP 执行上下文与分发、`2d9d7870` FTDMM 载体、`30457e3a` 测试；详细设计见 [EP 接入与执行设计](../design/onnx-support/ep-design.md) §6.3 与 §6.6。 |

<a id="adr-014"></a>

## ADR-014：框架前端与 addon 目录集中，构建按框架开关切分

| 项目 | 内容 |
| --- | --- |
| **状态** | accepted，2026-10-15（随文档迁移提交 `a88d3d09` 进入本基线，决策内容与当时实现一致） |
| **上下文** | 框架接入实现的位置与构建入口此前各自生长：TF 前端工具在 `annc/tools/`、TF addon 在 `tensorflow_addons/`，靠框架开关与 TensorFlow 的构建耦合隔离；ONNX 前端已按 `frontends/onnx/` 落点接入；而 ONNX Runtime 接入层完全在根构建之外——只能按 README 的手工命令编译，既没有落点也没有开关约定。后果是：接入实现游离于构建之外，无法用同一套开关做构建与安装验收；框架前端位置不对称（TF 在 `annc/tools/`，ONNX 在 `frontends/`）。 |
| **决策** | 目录按职责集中：框架前端入 `frontends/<framework>/`（本期 ONNX：`frontends/onnx/onnx2atir/` + `frontends/onnx/protos/`），框架接入 addon 入 `addons/<framework>/`（本期 `addons/onnxruntime/`）。根 CMake 引入四个开关 `ANNC_ENABLE_TENSORFLOW`、`ANNC_ENABLE_TENSORFLOW_ADDON`、`ANNC_ENABLE_ONNX`（默认 ON）、`ANNC_ENABLE_ONNXRUNTIME_EP`（默认 OFF）：父开关控制该框架的前端工具、proto 准备与依赖检查，子开关只控制 addon；addon 依赖父开关，冲突在 configure 期报错；EP 另需 `ANNC_ONNXRUNTIME_INCLUDE_DIR` 指向外部 ORT 1.26.0 头文件目录，缺失即失败、不下载 SDK。`build.sh` 提供 `--disable-tensorflow` / `--disable-tensorflow-addon` / `--disable-onnx` / `--enable-ort-ep` 与 `--regen-onnx-protos`，并把新值纳入缓存一致性检查（旧缓存缺键时：请求值等于旧默认组合则就地重配，改变默认组合要求 `--clean`）。**TF 源码、proto 路径与公共接口本期不迁移**，只改其构建入口。 |
| **后果** | ✅ 单框架可独立构建与验收（ONNX-only 不需要 TensorFlow/Python TF，core-only 不需要 protobuf）；✅ 新增框架前端有固定落点（`frontends/CMakeLists.txt` 追加）与统一开关约定；✅ ORT EP 有了默认关闭的可选构建目标，默认构建不引入 ORT SDK 依赖；✅ 默认配置与改动前等价（工具名、`build/bin`、`install/bin` 位置与安装目标不变）。❌ 顶层多一层目录与一维开关，旧构建缓存的首次使用需要一次就地重配或 `--clean`；❌ README/AGENTS/架构文档/Jenkins 需要维护开关矩阵；❌ TF 源码仍在原处，`frontends/` 暂时只装 ONNX，两套前端位置不对称（待 TF 迁移方案另立）。 |
| **备选方案** | (a) 一次性把 TF 前端一并迁入 `frontends/tensorflow/`——改动面覆盖 proto 生成路径与三个 TF 工具，风险与评审成本高于本期目标，拆分到后续方案；(b) 保持现状只加开关——目录职责仍不清晰，新增前端无落点，且 ORT EP 继续游离于构建之外；(c) 抽象 `cmake/AnncFrontendProto.cmake` 统一两套 proto 生成——只有 ONNX 一侧可迁移，收益不足，待 TF 到来再评估。 |
| **源码依据** | 根 [CMakeLists.txt](../../CMakeLists.txt)（`ANNC_ENABLE_TENSORFLOW`、`ANNC_ENABLE_TENSORFLOW_ADDON`、`ANNC_ENABLE_ONNX`、`ANNC_ENABLE_ONNXRUNTIME_EP`、`ANNC_ONNXRUNTIME_INCLUDE_DIR` 及开关冲突检查）、[frontends/CMakeLists.txt](../../frontends/CMakeLists.txt)、[addons/onnxruntime/CMakeLists.txt](../../addons/onnxruntime/CMakeLists.txt)、[build.sh](../../build.sh) |

<a id="adr-015"></a>

## ADR-015：ONNX 共享编译器与 ORT 插件接入的整体方案

| 项目 | 内容 |
| --- | --- |
| **状态** | proposed，2026-10-12；待架构评审后明确接受条件 |
| **继承与新增** | 本条目**继承**以下已接受（accepted／implemented）的决策，不因本条目为 proposed 而改变其状态：三层架构与框架无关优化层（ADR-002）、ONNX 前端“不可表达即降级”契约（ADR-011）、由 pipeline 报告驱动申报与运行时编译缓存（ADR-012）、执行上下文 ABI v3 与线程池入上下文（ADR-013）、`frontends/` 与 `addons/` 目录及构建开关（ADR-014）。**新增待接受约束**限于整体方案层面的部分：首期发布范围与模式清单、缓存身份的保守方案与命名空间升级规则、发布门槛（正确性风险关闭与真实宿主验证），以及整体方案 §12.2 列出的待评审决策项。 |
| **上下文** | ONNX 前端、公共 ATIR 融合、ONNX pipeline 和 ORT EP 已有主链路，但导入支持不等于可交付执行支持。缓存身份未覆盖 initializer 内容、EP 契约校验仍有缺口、多会话与多实例并发行为缺少验证证据等风险需要在整体方案中收敛（现状见整体方案 §12.1 与模块详细设计的差距章节）。 |
| **建议方案** | 保持既有三层架构：ONNX 专属语义适配复用公共 Builder，公共 ATIR Pass 决定融合，独立 ORT EP 负责申报和执行。延续会话初始化编译及磁盘缓存，首期按验证后的 AOT 模式、类型、形状和 ABI 发布。 |
| **建议约束** | 接管前检查完整区域和接口；当前执行失败不承诺 CPU 子图重试。缓存先纳入模型及全部常量语义、EP/工具链/kernel 库身份、目标和配置，再评估选择性排除权重。可靠总超时、版本校验和端到端差分为发布条件。 |
| **后果** | 复用既有优化和后端，避免 ONNX 专属运行时及重复匹配器；首次会话编译和完整常量摘要增加初始化成本。正确性风险关闭与真实宿主测试是发布前工作，不因方案通过而视为已完成。 |
| **备选方案** | 离线模型改写及 manifest 分发需额外产物协议，首期不恢复；独立执行整个模型的运行时覆盖与维护成本过大，不采用。 |
| **评审输出** | 首期版本和模式清单、ABI 与动态形状条件、缓存格式及升级规则、CPU 会话切换方案、资源与性能预算，以及风险关闭责任和退出条件。 |
| **详细方案** | [ANNC ONNX 支持整体方案设计](../design/onnx-support/overall.md)，重点见第 7、10、12 节。 |
