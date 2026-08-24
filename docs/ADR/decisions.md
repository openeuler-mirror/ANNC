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
| **决策** | `ANNCOptimizer` 在 `ANNC_JIT_ENABLE=1` 时让 pipeline 只保留 fusion-only ATIR，并将其路径写入 `ANNCFused`。`ANNCFusedOp::Compute` 提取实际 shape，同步调用 `annc-asm` 和 `annc`，加载生成的共享库并执行 kernel。本阶段不引入编译缓存、异步编译或运行时 fallback。 |
| **后果** | 动态 shape 的编译决策延后到运行时且边界清晰；首次调用包含编译开销，每次调用都可能重新生成共享库，缓存和并发控制留待后续独立提交。 |
| **备选方案** | 在 Grappler 阶段按样例 shape 预编译——无法覆盖运行时动态 shape；异步 JIT——需要额外的请求排队和失败语义。 |

<a id="adr-008"></a>

## ADR-008：TensorFlow 进程内 JIT 编译缓存

| 项目 | 内容 |
| --- | --- |
| **上下文** | ADR-007 的同步 JIT 能将请求实际 shape 交给后端，但每次调用都重新执行 `annc-asm`、`annc` 和 `dlopen`。直接按 `kernel_name` 缓存会阻止语义相同但来源名称不同的 fusion func 复用。工具链、GEMM 配置和 ABI 是进程级固定上下文，不需要在每次推理时重新探测。 |
| **决策** | 在 `ANNCFusedOp` 进程内增加有界编译缓存。key 由 cache schema version、与名字无关的 canonical ATIR template fingerprint 和按 kernel 参数顺序排列的实际 shape 构成；Grappler 阶段将影响 codegen 的 module 属性（包括 `annc.intra_thread_count`）纳入 template fingerprint。`kernel_name` 和 MLIR func 名称不进入 key，只在 miss 时选择 func 和解析首次生成的符号。相同 key 使用 single-flight，同步等待同一编译结果；不同 key 可并行编译。成功 entry 以 `shared_ptr` 管理 `dlopen` handle 和工作目录并按 LRU 淘汰，默认上限 64，`ANNC_JIT_CACHE_MAX_ENTRIES=0` 可关闭缓存。失败结果从表中移除，后续请求重新编译。运行时 shape JSON 只携带参数索引和实际 shape，dtype 由 ATIR 模板和 TensorFlow Op dtype contract 共同约束。 |
| **后果** | 相同语义和 shape 在进程内只编译一次；工具链、GEMM 配置和 ABI 变化属于不支持的进程级上下文切换，需要重启进程；正在执行的产物在淘汰后仍存活到最后一个引用释放。首次请求仍承担同步编译延迟，缓存仅限当前进程。 |
| **备选方案** | (a) 每个 Op 仅缓存最后一个 shape——无法跨节点复用且 shape 切换会重复编译；(b) 使用 `kernel_name` 作为 key——把来源身份误当成代码生成语义；(c) 跨进程磁盘缓存——需要额外的原子发布、完整版本校验和不可信 `.so` 安全边界，本阶段不采用。 |
