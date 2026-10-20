# ANNC 文档

本目录是 ANNC 文档的唯一来源。文档与源码同仓、同评审、同发布；导航由**目录结构 + 各级 `README.md`** 承担（不使用静态站点生成器）。

> **迁移状态**：架构、设计、ADR 与 ONNX 文档集均已迁入新目录结构；原 `docs/Feature/`、`docs/ADR/` 目录已移除。目标结构与迁移计划见内部提案 `docs/superpowers/docs-restructure-proposal.md`（该目录已 gitignore）。

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
| [getting-started/](getting-started/README.md) | 安装与快速开始（教学） | 部分就位 |
| [guides/](guides/README.md) | 面向任务的操作指南 | 部分就位 |
| [architecture/](architecture/README.md) | 总体架构与概念解释 | 已就位 |
| [design/](design/README.md) | 特性与模块详细设计（HLD/LLD） | 已就位 |
| [decisions/](decisions/README.md) | 架构决策记录（ADR） | 已就位 |
| [rfcs/](rfcs/README.md) | 待决设计提案 | 规划中 |
| [contributing/](contributing/README.md) | 开发环境、测试、规范与流程 | 规划中 |
| [assets/](assets/README.md) | 图片与图表源文件 | 已就位 |

## 迁移状态

历史文档已全部迁入新结构：架构总览（[architecture/](architecture/README.md)）、JIT 常量折叠与 ONNX 支持（[design/](design/README.md)）、架构决策（[decisions/](decisions/README.md)）。原 `docs/Feature/`、`docs/ADR/` 目录已移除。

用户文档开始落位：安装（[getting-started/installation.md](getting-started/installation.md)），从源码构建、CLI 工作流、环境变量、Python 绑定与排障（[guides/](guides/README.md)），运行时接入（[architecture/runtime-integration.md](architecture/runtime-integration.md)）。仓库根 `README.md` 已收敛为项目名片，只保留定位、支持范围与接入前提、快速开始、按角色导航、架构速览、工具矩阵与相关项目。

## 编写约定

- 命名、元数据、模板与文档类型边界见 [contributing/README.md](contributing/README.md)。
- 项目级开发约定以仓库根目录 [AGENTS.md](../AGENTS.md) 为唯一来源。
