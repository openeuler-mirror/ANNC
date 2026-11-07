# AArch64 GEMM Codegen 设计

| 项目   | 内容                                                             |
| ---- | -------------------------------------------------------------- |
| 文档编号 | ANNC-DESIGN-GEMM-001                                           |
| 状态   | 已实现；描述当前源码行为                                                   |
| 责任人  | ANNC AArch64 后端维护者                                             |
| 源码基线 | `origin/feat/gemm-epilogue-codegen` `8c4230e` + 当前 GEMM 策略重构分支 |

## 目标与范围

本文说明 AArch64 GEMM 从问题识别、策略配置、线程规划到 microkernel 调用生成的当前设计。它面向后端开发者，解释阶段职责、元数据流和已知边界，不定义新的调优算法或用户配置格式。Epilogue 二进制 codegen 契约见 [GEMM epilogue 设计](gemm-epilogue-codegen.md)。

当前实现面向 AArch64 F32 GEMM，支持 NEON / SVE，以及 GEMM、matrix-vector、vector-matrix 执行路径。cache 和 kernel tile 策略目前都只有 fixed 类型；策略选择现通过阶段专用的 tiling policy 接口生成压缩计划，再由 pass 通用物化。此接口为后续 tiling 策略提供扩展点，但不代表已有自动调参或多种 tiling 算法选择。

## 设计概览

```text
GEMM op
  -> extract problem
  -> select execution / packing path + load tuning strategy
  -> plan static thread topology
  -> thread tiling -> cache blocking -> kernel tiling
  -> materialize leaves -> select concrete microkernel
  -> epilogue lowering -> microkernel lowering -> ABI lowering
```

| 阶段       | 输入与结果                                                   | 主要职责                                                                                                                                                           |
| -------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 问题提取     | GEMM op -> `gemm_problem`                               | 记录 M/N/K 与 leading dimensions；不选择 tile。                                                                                                                        |
| 策略选择     | 问题 + tuning config -> `GemmThreadPlanningInput`         | 选择执行路径和 RHS packing；携带 target、ISA、数据类型、cache/kernel 策略、代价模型与线程预算。带 epilogue 的不兼容专用路径回退到通用 GEMM。                                                                |
| 线程规划     | planning input -> `GemmCandidate`                       | 以实际 microkernel MR/NR 为工作粒度，估算可选任务网格、任务数和 task ID 映射顺序。M 任务按 MR 对齐；N 任务按 16 个 F32 列对齐。详细规则见[线程切块设计](aarch64-gemm-thread-tiling.md)。                            |
| tiling   | `GemmProblem` + planned `GemmCandidate` -> tiled Linalg | 线程 tiling 消费任务 topology；cache/kernel tiling 读取 strategy reference，并调用对应策略接口计算 tile。策略接口生成压缩 tile plan，tiling pass 负责展开 plan；fixed 策略以 tile run 描述等尺寸重复块，不逐块存储。 |
| leaf 与选核 | 静态 microtile + strategy reference -> leaf call          | leaf materialization 读取 problem 的 leading dimensions 和实际 M/N/K；final selector 从 strategy reference 读取 ABI 选择，按实际 shape 选择具体 NEON/SVE 符号并检查范围。                  |
| lowering | 已选 leaf -> ABI call                                     | epilogue lowering 消费 leaf 上的选择与 epilogue carrier；microkernel lowering 绑定普通或生成符号；ABI lowering 物化调用参数。                                                           |

标准 pipeline 顺序由 `Aarch64CodegenPipelineBuild.cpp` 维护。RHS packing / prepacking 是与主体 tiling 配合的独立数据路径；预打包 binding 单独附着在 GEMM op 上。
路径选择调用 kernel 策略的 `configureForExecution` 取得该执行路径的参数，再由策略校验 kernel ABI；fixed 策略对 MatVec / VecMat 使用对应 ABI 的最大 MR 和 panel lanes。策略不支持专用路径时重新选择通用 GEMM 路径并计算 RHS packing；epilogue 的路径限制保留原有 direct packing。读取 metadata 时复用策略校验，选择 pass 不直接读取 fixed 参数。

## 策略与规划数据

调优 JSON v3 提供 target、ISA、数据类型、cache/kernel 策略选择及其策略专属参数，并提供 thread planner cost model。每个策略先由 `strategy` 字段选择，再从同级同名对象读取参数：`cache.strategy` 对应 `cache.fixed`，`kernel.strategy` 对应 `kernel.fixed`。通用配置解析位于 `Support/GemmConfig.cpp`；策略参数的 JSON 与 MLIR metadata 编解码位于 `Support/GemmStrategyConfig.cpp`。当前 cache/kernel 策略 kind 都只有 `fixed`；cost model 用于比较线程任务拓扑，不会动态搜索 microkernel tile。

策略配置字段示例（省略通用字段）：

```json
{
  "cache": {"strategy": "fixed", "fixed": {"mc": 128, "nc": 256, "kc": 128}},
  "kernel": {"strategy": "fixed", "fixed": {"mr": 4, "panel_lanes": 1}}
}
```

四类数据分别承载不同阶段的信息：

- `GemmProblem` 描述算子问题和内存步长。
- `GemmTuningStrategy` 描述可复用的目标相关选择，不包含具体问题尺寸或线程预算。
- `GemmThreadPlanningInput` 将 strategy、cost model、最大线程数交给 planner。
- `GemmCandidate` 保存 strategy reference 和 planner 得出的静态任务网格。`GemmProblem` 与 candidate 随 GEMM op 传递；不再生成跨阶段的 finalized GEMM plan。

Cache 与 kernel tiling 是两个独立的配置维度，因此分别由 `GemmCacheTilingStrategy` 和 `GemmKernelTilingStrategy` 提供阶段接口。它们使用各自的请求和计划类型，不强行合并成一个同时包含两套配置的策略对象。fixed 配置分别由 `FixedCacheParameters` 和 `FixedKernelParameters` 承载，并通过策略参数变体保存。Cache pass 向 cache policy 提交上游 `GemmBlockShape`，取得行、列分区及带 KC 角色的 reduction runs。Kernel pass 提交上游 block shape 和 `GemmKernelTargetInfo`，策略自行推导 NR 并返回有序的 `GemmKernelTileGroup` 列表。每组包含一个外层 tile run 和多个有序内层 tile run；run 记录 offset、tile 大小及重复次数。pass 按组物化外层循环，在同一个外层迭代内依次物化各内层 run。策略算法只在对应 policy 实现中，pass 只负责物化返回结构。

当前 fixed kernel tiling 为每段连续行块生成一个 rows-first group；同一行块内依次执行完整列块和尾列块。UniGEMM 专属的 pbg/pb 角色尚未加入 fixed 策略接口。

运行时 packed-B workspace 在每个函数入口分配并复用。leaf pass 汇总函数内各 RHS cache block 的静态 K/N 和 ABI 对齐后的容量需求：NEON 取最大 packed 元素数，SVE 的动态容量辅助函数接收最大 K/N。这样尾列块仍按 pack 布局占用完整对齐宽度，同时保持同一函数内的单份 workspace。

中间 `problem`、`candidate` 属性附着在 GEMM op。strategy kind 和参数存入所在函数的 `annc.aarch64.gemm_strategies` 字典表，candidate 只保留稳定引用和线程网格。strategy table 将通用策略标识与 `cache_parameters` / `kernel_parameters` 两个策略专属字典分开；添加策略参数时由策略 codec 处理，不要求通用 metadata writer 枚举参数字段。相同字典值在函数内 intern 复用；清理阶段会移除不可达项。

## Tile 到 microkernel 的关系

kernel tile 是策略参数，thread planning 用它推导任务粒度，kernel tiling 调用策略接口产生静态子块。leaf materialization 从每个子块得到实际 M/N/K，写入 leaf 的 discardable 属性。`selectFinalMicrokernel` 同时读取这些实际维度与 strategy：

- 用实际 M/N 与 kernel ABI 的最大支持范围检查 microtile；tile 策略参数不限制 selector 的输入 shape。
- SVE 根据实际 M、N 和向量长度计算最终 N-vector group。
- NEON 根据实际 M/N/K、ABI 的 K unroll 与累加模式构造符号变体。
- 根据 execution kind 和 RHS packing 校验 leaf ABI，并区分 GEMM、MatVec、VecMat 路径。

因此，策略改变 tile 参数会沿 candidate 的 strategy reference 和 tiling 结果影响最终 kernel 符号。selector 不重新选择策略；当选择规则或 kernel ABI 改变时，才需要同步修改此层及相关测试。

## 不变量与边界

- GEMM op 上的 candidate 指向唯一 strategy；后续阶段通过该引用读取选择，并由对应 tiling strategy 计算 tile。
- thread planner 输出的任务数、分片方向和 tile 对齐必须与 kernel ABI 一致；单线程预算可保持串行执行。
- KC 首块使用 overwrite，后续块使用 accumulate；final-K epilogue 只在 leaf 阶段确定插入位置。
- 当前 tuning tile 类型只有 fixed；新增策略类型需贯通配置解析、策略实现、必要的线程规划约束和测试。cache/kernel tiling pass 消费相应接口生成的压缩结果，不应再增加按策略种类分叉的切块算法。
- 当前 RHS 预打包布局依赖 fixed cache 与 kernel 参数，因此只有两者都选择 fixed 时才允许预打包；不兼容组合在策略选择或 metadata 解码时以 codegen error 拒绝。
- 支持范围以源码与测试为准。当前实现没有自动 benchmark/autotuning，也不承诺非 AArch64、非 F32 或未声明的 microkernel ABI。

## 验证

FileCheck 覆盖问题提取、strategy/candidate metadata、cache/thread/kernel tiling、NEON/SVE leaf ABI、packing 与完整 AArch64 pipeline。`tests/unit/kernels/gemm_thread_planner_test.cpp` 覆盖线程拓扑代价规划。改动策略或 leaf 选择时，应同时验证 strategy metadata 和最终 leaf 符号 / ABI；涉及数值语义时另加目标平台数值验证。
