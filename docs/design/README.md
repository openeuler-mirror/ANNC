# 特性与模块设计（工程文档）

承载特性的整体方案（HLD）与模块详细设计（LLD）。一个特性一个子目录，HLD 与 LLD 就近维护。

## 现有文档

| 文档集 | 当前位置 | 计划落点 |
| --- | --- | --- |
| JIT 常量折叠特性设计 | [../Feature/jit-constant-folding-architecture-design.md](../Feature/jit-constant-folding-architecture-design.md) | `jit-constant-folding/` |

> ONNX 文档集（前端 / 融合 / EP）后续单独迁入本目录的 `onnx-support/`。

## 约定

- 设计文档回答「是什么 / 怎么做」；决策理由见 [../decisions/](../decisions/README.md)。
- 文档须声明编号、状态、责任人与源码基线，见 [../contributing/](../contributing/README.md)。
