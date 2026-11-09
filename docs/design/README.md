# 特性与模块设计（工程文档）

承载特性的整体方案（HLD）与模块详细设计（LLD）。一个特性一个子目录，HLD 与 LLD 就近维护。

## 现有文档

| 文档集 | 位置 |
| --- | --- |
| AArch64 GEMM（codegen、线程切块、epilogue） | [gemm/](gemm/README.md) |
| JIT 常量折叠早期提案（历史记录；当前实现见右列） | [jit-constant-folding/overview.md](jit-constant-folding/overview.md) |
| TensorFlow JIT 代码与预打包权重分离实现契约 | [jit-constant-folding/cache-separation.md](jit-constant-folding/cache-separation.md) |
| ONNX 支持（整体方案 + 前端 / 融合 / EP 详设） | [onnx-support/](onnx-support/README.md) |
| TensorFlow 1.15 支持（待评审） | [tensorflow-1-15-support/overall.md](tensorflow-1-15-support/overall.md) |
| ANNC 异构图切分优化器（待评审，已搭建代码骨架） | [总体设计](annc-partition/overview.md) · [CLI 入口详细设计](annc-partition/cli-design.md) |
| Kernel Context 设计与测试 | [Kernel Context 与统一 Runtime ABI 设计](runtime/kernel-context-runtime-design.md) |

## 约定

- 设计文档回答「是什么 / 怎么做」；决策理由见 [../decisions/](../decisions/README.md)。
- 文档须声明编号、状态、责任人与源码基线，见 [../contributing/](../contributing/README.md)。
