# 架构（解释）

回答「系统长什么样、为什么这样」。跨特性复用的概念与全景放这里；单个特性的设计放 [../design/](../design/README.md)。

## 现有文档

| 文档 | 位置 |
| --- | --- |
| ANNC 总体架构设计 | [overview.md](overview.md) |
| 运行时接入（TensorFlow / ONNX Runtime） | [runtime-integration.md](runtime-integration.md) |

## 规划内容

- `compilation-pipeline.md`：ATIR → lowering → AArch64 代码生成的编译流水线
- `quality-attributes.md`：可靠性、安全、性能等非功能质量属性
