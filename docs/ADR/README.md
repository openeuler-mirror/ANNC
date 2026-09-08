# ANNC 架构决策记录（ADR）

## 什么是 ADR

ADR（Architecture Decision Record）用于记录对 ANNC 架构有长期影响的决策，包括决策的上下文、所做的选择、产生的后果以及当时放弃的备选方案。

## 何时添加 ADR

当变更涉及以下范围时，应新增 ADR：

- 新增 dialect、backend 或 frontend；
- 重大 lowering 策略或编译流水线变更；
- 关键接口契约变更（NodeInfo / Fusion Metadata / Kernel C ABI）；
- 进程模型、安全模型、配置 schema 等重大设计选择。

以下情况不需要 ADR：

- 日常 bugfix；
- 不影响架构的纯实现细节优化；
- 已有 ADR 的微小调整。

## ADR 状态

- **proposed**：已提出，尚未达成共识；
- **accepted**：已接受，项目按此决策执行；
- **deprecated**：已过时，但保留以供查阅；
- **superseded**：被新的 ADR 取代，需在条目中注明替代者。

## ADR 索引

| 编号 | 标题 | 状态 | 详情位置 |
| --- | --- | --- | --- |
| ADR-001 | 采用统一 IR（ATIR）而非框架专属优化 | accepted | [decisions.md#adr-001](decisions.md#adr-001) |
| ADR-002 | 编译器三层架构（ANNC 框架对接层 / ANNC 工具链前端 / ANNC 工具链后端） | accepted | [decisions.md#adr-002](decisions.md#adr-002) |
| ADR-003 | 两层选择算子接入策略 | accepted | [decisions.md#adr-003](decisions.md#adr-003) |
| ADR-004 | fork/exec 进程隔离 | accepted | [decisions.md#adr-004](decisions.md#adr-004) |
| ADR-005 | 运行时无 fallback 决策 | accepted | [decisions.md#adr-005](decisions.md#adr-005) |
| ADR-006 | tf2atir 使用已解析 TF 图作为 NodeInfo 前置边界 | accepted | [decisions.md#adr-006](decisions.md#adr-006) |
| ADR-007 | TensorFlow 无缓存同步 JIT | accepted | [decisions.md#adr-007](decisions.md#adr-007) |
| ADR-008 | TensorFlow 进程内 JIT 编译缓存 | accepted | [decisions.md#adr-008](decisions.md#adr-008) |
| ADR-009 | GEMM intra 线程数改为运行时 JIT 特化维度 | accepted | [decisions.md#adr-009](decisions.md#adr-009) |
| ADR-010 | TF 控制流镜像转换（mirror，不做语义重建） | accepted | [decisions.md#adr-010](decisions.md#adr-010) |

新增 ADR 时，请在 `decisions.md` 中以如下格式添加详情，并在此索引表末尾追加一行：

```markdown
<a id="adr-NNN"></a>

## ADR-NNN：标题
```
