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
| ADR-002 | 编译器三层解耦（前端/优化/运行时） | accepted | [decisions.md#adr-002](decisions.md#adr-002) |
| ADR-003 | 两层选择算子接入策略 | accepted | [decisions.md#adr-003](decisions.md#adr-003) |
| ADR-004 | fork/exec 进程隔离 | accepted | [decisions.md#adr-004](decisions.md#adr-004) |
| ADR-005 | 运行时无 fallback 决策 | accepted | [decisions.md#adr-005](decisions.md#adr-005) |

新增 ADR 时，请在 `decisions.md` 中以如下格式添加详情，并在此索引表末尾追加一行：

```markdown
<a id="adr-NNN"></a>

## ADR-NNN：标题
```
