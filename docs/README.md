# ANNC 文档

本目录是 ANNC 文档的唯一来源。文档与源码同仓、同评审、同发布；导航由**目录结构 + 各级 `README.md`** 承担（不使用静态站点生成器）。

> **迁移进行中**：新目录结构正在搭建，现有文档暂留在原路径，正文链接指向当前位置。目标结构与迁移计划见内部提案 `docs/superpowers/docs-restructure-proposal.md`（该目录已 gitignore）。

## 按受众的阅读路径

| 你是谁 | 建议入口 |
| --- | --- |
| 第一次使用 | [getting-started/](getting-started/README.md) |
| 要把模型接进框架 | [guides/](guides/README.md) |
| 要理解整体设计 | [architecture/](architecture/README.md) |
| 要改编译器 | [design/](design/README.md) · [contributing/](contributing/README.md) |
| 要查历史决策 | [decisions/](decisions/README.md) |

## 目录导航

| 目录 | 内容 | 状态 |
| --- | --- | --- |
| [getting-started/](getting-started/README.md) | 安装与快速开始（教学） | 规划中 |
| [guides/](guides/README.md) | 面向任务的操作指南 | 规划中 |
| [architecture/](architecture/README.md) | 总体架构与概念解释 | 迁移中 |
| [design/](design/README.md) | 特性与模块详细设计（HLD/LLD） | 迁移中 |
| [decisions/](decisions/README.md) | 架构决策记录（ADR） | 迁移中 |
| [rfcs/](rfcs/README.md) | 待决设计提案 | 规划中 |
| [contributing/](contributing/README.md) | 开发环境、测试、规范与流程 | 规划中 |
| [assets/](assets/README.md) | 图片与图表源文件 | 已就位 |

## 现有文档（迁移前位置）

| 文档 | 当前位置 |
| --- | --- |
| ANNC 总体架构设计 | [annc-architecture-design.md](annc-architecture-design.md) |
| JIT 常量折叠特性设计 | [Feature/jit-constant-folding-architecture-design.md](Feature/jit-constant-folding-architecture-design.md) |
| 架构决策记录（ADR） | [ADR/README.md](ADR/README.md) |

> ONNX 文档集（前端 / 融合 / EP）单独迁移，暂不在此列。

## 编写约定

- 命名、元数据、模板与文档类型边界见 [contributing/README.md](contributing/README.md)。
- 项目级开发约定以仓库根目录 [AGENTS.md](../AGENTS.md) 为唯一来源。
