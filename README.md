# ANNC (Accelerated Neural Network Compiler)

## 目录

- [项目简介](#项目简介)
- [主要特性](#主要特性)
- [构建环境和依赖](#构建环境和依赖)
- [构建与使用](#构建与使用)
- [版本历史](#版本历史)
- [许可证](#许可证)
- [贡献声明](#贡献声明)

## 项目简介

ANNC (Accelerated Neural Network Compiler) 是一个基于 OpenXLA/XLA 框架的 AI 编译器，专注于 CPU 平台的神经网络推理优化。它在提供跨平台通用优化的基础上，特别针对 ARM 架构（尤其是鲲鹏处理器）进行了深度定制和性能调优，通过图融合、算子优化和常量折叠等技术，显著提升模型在目标平台上的推理性能。

## 主要特性

| 特性名称 | 说明 | 特性启用 |
|---------|------|----------|
| TensorFlow图融合 | 提供TensorFlow模型层面的图融合与图重写功能，并在后端提供自定义融合算子的手动实现 | `annc-opt -I input_model.pb -O output_dir embed_hash_bucket`<br>`export ENABLE_BISHENG_GRAPH_OPT=""` |
| XLA图融合 | 提供ANNC XLA图融合特性，识别并消除融合后的重复计算 | `export ANNC_FLAGS="--graph-opt"` |
| 算子优化 | 提供ANNC算子优化特性，将GEMM算子和Softmax算子通过模式匹配接入OpenBLAS和XNNPACK高性能后端 | `export ANNC_FLAGS="--gemm-opt"`<br>`export XLA_FLAGS="--xla_cpu_enable_xnnpack=true"` |
| 常量折叠 | 提供GEMM算子常量折叠优化，消除运行时数据重排开销（[详细说明](docs/constant-folding.md)） | `annc-opt -I input_model.pb -O output_dir layout_matmul`<br>+ `export ANNC_FLAGS="--layout-matmul"` |

## 构建环境和依赖

- **Bazel**: 6.5.0
- **Python**: 3.x
- **GCC/G++**: 支持 C++17 或更高版本
- **依赖库**:
  - OpenXLA/XLA (commit: 40008cb2c85749ae436be61c40d3279cb24705c7)
  - OpenBLAS
  - XNNPACK

## 构建与使用

详细的构建和测试流程请参考 [ANNC 测试指南](TESTING.md)。

### 构建指南

#### 构建共享库

```bash
bazel --output_user_root=./output build -c opt annc/service/cpu:libannc.so
```

#### 构建静态库

```bash
bazel --output_user_root=./output build -c opt annc/service/cpu:annc
```

#### 构建测试工具

```bash
bazel --output_user_root=./output build -c opt annc/tools/kp-opt:kp-opt
bazel-bin/annc/tools/kp-opt/kp-opt {test_hlo_cluster_file.dat}
```

#### 调试模式构建

```bash
bazel --output_user_root=./output build -c opt --copt="-g" annc/service/cpu:libannc.so
```

#### 使用本地缓存目录

```bash
bazel --output_user_root=./output --distdir=/path/to/proxy build -c opt annc/service/cpu:libannc.so
```

### 使用方法

#### Python API

提供 annc-opt 工具对 PB 模型进行预编译，配合 ANNC 编译优化，实现图优化和常量折叠特性。

annc-opt 工具在模型部署前对 TensorFlow SavedModel 进行预处理，执行模式匹配和图重写操作，将复杂的算子组合融合为高效的自定义算子；同时在编译阶段通过常量折叠技术预先计算静态张量，减少运行时开销，提升推理性能。


```bash
# 工具安装
cd /path_to_ANNC/python
python3 setup.py bdist_wheel
python3 -m pip install dist/*.whl --force-reinstall

# 模型转换
# 常量折叠预重排打包
annc-opt -I input_model.pb -O output_dir layout_matmul
# TensorFlow图融合
annc-opt -I input_model.pb -O output_dir embed_hash_bucket
```

#### 特性开关配置

ANNC 提供了丰富的特性开关，通过环境变量 `ANNC_FLAGS` 进行配置，可以精细控制各种优化策略的启用。

| 开关名称 | 类型 | 说明 | 默认值 |
|---------|------|------|--------|
| `--gemm-opt` | 组合标志 | **启用所有 GEMM_OPT 类型优化**（包括 matmul、layout-matmul、disable-tf-matmul-fusion） | false |
| `--graph-opt` | 组合标志 | **启用所有 GRAPH_OPT 类型优化**（包括 sps-emd-2） | false |
| `--matmul` | GEMM_OPT | 注册标准 matmul 算子（使用 DNN_CUSTOM_CALL） | false |
| `--layout-matmul` | GEMM_OPT | 注册 layout_matmul 算子，支持自定义内存布局（使用 FUSED_OPERATION） | false |
| `--disable-tf-matmul-fusion` | GEMM_OPT | 禁用 TensorFlow 层面的 matmul 融合，交由 ANNC 处理 | false |
| `--batch-matmul` | NONE | 注册 batch_matmul 算子，支持批量矩阵乘法 | false |
| `--matmul-add` | NONE | 注册 matmul_add 融合算子（MatMul + Add） | false |
| `--matmul-add-relu` | NONE | 注册 matmul_add_relu 融合算子（MatMul + Add + ReLU） | false |
| `--sps-emd-2` | GRAPH_OPT | 启用 sparse_embedding2 融合优化，优化稀疏嵌入操作 | false |
| `--pooling` | NONE | 启用 pooling 融合优化，将多个操作融合为单个 pooling 操作 | false |
| `--annc-pass` | TF_OPT | 启用 ANNC 优化器 pass，对 TensorFlow 计算图进行优化 | **true** |

##### 使用示例

```bash
# 启用所有 GEMM 和图优化
export ANNC_FLAGS="--gemm-opt --graph-opt"

# 启用特定的 GEMM 优化
export ANNC_FLAGS="--layout-matmul --matmul-add-relu"

# 只启用 matmul
export ANNC_FLAGS="--matmul"
```

## 版本历史

| 版本 | 发布日期 | 新增特性 |
|------|----------|----------|
| v0.0.4 | 2026-05 | 常量折叠优化 |
| [v0.0.3](https://gitcode.com/openeuler/ANNC/-/tree/v0.0.3) | 2025-11 | TensorFlow图融合（9个Embedding融合算子）、算子优化（Kernel Selector智能路由 + XNNPACK集成） |
| [v0.0.2](https://gitcode.com/openeuler/ANNC/-/tree/v0.0.2) | 2025-08 | XLA图融合（CPU感知图编译增强、多核搜索系统）、算子优化（ENABLE_ANNC编译控制） |
| [v0.0.1-alpha](https://gitcode.com/openeuler/ANNC/-/tree/v0.0.1-alpha) | 2025-05 | XLA图融合（XLA/LLVM Patch机制）、算子优化（GEMM/Softmax接入OpenBLAS） |

## 许可证

- 本项目核心代码采用 Apache License 2.0
- third_party/kpgemm 采用 BSD 3-Clause License (OpenBLAS)


## 贡献声明

欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交[Issues](https://gitcode.com/boostkit/community/blob/master/docs/contributor/issue-submit.md)联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎大家在[讨论专区](https://gitcode.com/boostkit/community/discussions)展开讨论交流。感谢您的支持。
