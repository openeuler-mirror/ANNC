# 架构（解释）

回答「系统长什么样、为什么这样」。跨特性复用的概念与全景放这里；单个特性的设计放 [../design/](../design/README.md)。

## 现有文档

| 文档 | 当前位置 | 计划落点 |
| --- | --- | --- |
| ANNC 总体架构设计 | [../annc-architecture-design.md](../annc-architecture-design.md) | 本目录 `overview.md` |

## 规划内容

- `compilation-pipeline.md`：ATIR → lowering → AArch64 代码生成的编译流水线
- `runtime-integration.md`：TensorFlow / ONNX Runtime 两种接入方式
- `quality-attributes.md`：可靠性、安全、性能等非功能质量属性
