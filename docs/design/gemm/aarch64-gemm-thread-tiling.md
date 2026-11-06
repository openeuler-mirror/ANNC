# AArch64 GEMM 线程切块设计

| 项目 | 内容 |
| --- | --- |
| 文档编号 | ANNC-DESIGN-GEMM-002 |
| 状态 | 已实现；描述当前源码行为 |
| 责任人 | ANNC AArch64 后端维护者 |
| 源码基线 | 当前 GEMM 策略重构分支，基于 `origin/feat/gemm-epilogue-codegen` `8c4230e` |

## 范围

本文说明 AArch64 GEMM 静态线程网格的选择、任务范围切分和 task ID 映射。线程规划发生在 GEMM cache blocking 与 kernel tiling 之前；它确定任务网格，不物化 cache tile 或 kernel tile。整体阶段关系见 [AArch64 GEMM codegen 设计](aarch64-gemm-codegen.md)。

实现位于 `Support/GemmThreadPlanner.cpp`，IR 物化位于 `GemmThreadTiling.cpp`。planner 的静态网格约束与代价模型测试位于 `tests/unit/kernels/gemm_thread_planner_test.cpp`。
当前代价模型仅支持 fixed cache 与 fixed kernel 策略。已注册且可解析的其他策略在规划入口发出告警，并记录单线程网格 `(threadCount, tasksM, tasksN) = (1, 1, 1)`；线程物化阶段保留原始 GEMM 串行执行。当前配置解析仅注册 fixed，未知策略仍在解析阶段报错。

## 线程网格选择

planner 枚举候选网格 `(tasksM, tasksN)`。候选必须满足：

- `tasksM * tasksN <= max_threads`；
- 每个维度都能切出正的非末任务范围；
- M 维每个非末任务至少有一个完整 `MR`，N 维每个非末任务至少有一个完整 `NR`；
- 所有任务范围按各自对齐粒度切分。

其中 `MR`、`NR` 是当前 kernel family 的有效 microkernel 尺寸。搜索上限分别由 `M / MR`、`N / NR` 和线程预算限制，至少保留一个任务。最终网格按以下优先级选择：

1. 估算最慢任务耗时更低；
2. 耗时相同则选任务数更少的网格；
3. 耗时和任务数都相同则优先 `tasksN` 较大的网格。

最慢任务估算使用 M、N 两维中最大任务范围计算，以计入末任务吸收余数后的负载不均。耗时模型估算 GEMM/GEMV 计算、数据读写、kernel 调用和运行时 RHS packing；另外按任务数计入 TensorFlow fixed-block `ParallelFor` 的 dispatch 开销。各项系数来自 tuning config 或当前运行时 dispatch 模型。线程池上限只约束候选任务数，不保证最终任务数一定等于线程池大小。

## Cost model

每个候选网格估算最慢任务的工作量，再加该网格的 dispatch 开销：

```text
candidateCost = taskCost(maxMTask, maxNTask) + dispatchCost(tasksM * tasksN)
```

`taskCost` 按 cache MC/NC/KC 分块累计。每个 cache 块的估算包含：

- **计算**：按 MR/NR microtile 数和向量组数估算 GEMM 工作量；GEMV 使用独立的 K 归约工作量。
- **读写**：估算 A/B 输入流量及 C 输出流量。K 累加块额外计入读取 C；读写系数按有效 kernel 内存成本计，不代表 DRAM 延迟。
- **微内核调用**：按 cache 块中的 microtile 调用数计费。
- **RHS packing**：仅 runtime pack 路径计入源数据、对齐后的 packed 数据和 pack 调用成本；direct 与 prepacked 路径不计此项。

配置中的系数为 `load_cycles_per_byte`、`store_cycles_per_byte`、`compute_cycle_scale`、`gemv_compute_cycle_scale`、`microkernel_call_cycles`、`pack_cycles_per_byte` 和 `pack_call_cycles`。它们用于比较候选网格的相对成本，代价模型不执行运行时测量或动态调度。dispatch 项当前对应 TensorFlow `ParallelFor` 的固定块模型：任务数 `Q > 1` 时为 `12000 + 3300 * Q` cycles，`Q <= 1` 时为 0。

## M/N 范围与对齐

每一维独立计算任务范围。给定维度长度 `D`、任务数 `T` 和对齐值 `A`：

```text
if T == 1:
    base = D
    tail = D
else:
    base = alignDown(floor(D / T), A)
    tail = D - (T - 1) * base

offset(task) = task * base
size(task)   = (task == T - 1) ? tail : base
```

当前对齐规则：

| 维度 | 对齐值 | 原因 |
| --- | --- | --- |
| M | `MR` | 非末任务边界落在完整 microkernel 行块上。 |
| N | 16 个 F32 列 | 一个 64 字节输出行 cache line；减少相邻 N 任务写入同一 cache line。 |

N 维任务范围使用固定的 16 列对齐；`NR` 用于限制 N 维候选任务数并估算 kernel 工作粒度。`T > 1` 时，若 `alignDown(floor(D/T), A) == 0`，该切分无效，候选网格会被跳过。末任务吸收维度余数，范围长度不小于 `base`。当 `T == 1` 时任务覆盖整个维度，范围长度等于 `D`。

## task ID 映射

网格确定后，`shardByColumns` 选择 task ID 遍历网格的顺序，不改变网格或各任务范围：

```text
if tasksN == 1:
    shardByColumns = false
else if tasksM == 1:
    shardByColumns = true
else:
    shardByColumns = (N 的非末任务范围 >= M 的非末任务范围)
```

两维任务范围相等时，优先沿 N 维线性化。任务坐标由 task ID 计算：

```text
shardByColumns == true:
    mTask = taskId % tasksM
    nTask = taskId / tasksM

shardByColumns == false:
    mTask = taskId / tasksN
    nTask = taskId % tasksN
```

随后分别用各维的 `offset(task)` 和 `size(task)` 计算线程子问题。这样每个网格坐标对应唯一的 `(M, N)` 区域，所有区域完整覆盖输出矩阵。

## 相关测试

`gemm_thread_planner_test.cpp` 覆盖范围截断和末任务余数、候选网格选择、窄形状与 packing 成本，以及多组 M/N 形状下输出区域恰好覆盖一次。线程策略或 cost model 变更时，应同时检查 planner 结果和 thread tiling 生成的范围映射。
