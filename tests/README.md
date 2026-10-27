# ANNC 测试目录

本目录是 ANNC 所有受跟踪测试的唯一归属地。生产目录（`annc/`、`frontends/`、
`addons/`、`tensorflow_addons/`）只放实现，不得包含测试源码、GTest/Python
测试发现、`gtest_discover_tests` 或 `add_test` 注册；测试到生产目录只允许
单向依赖。

## 整体端到端测试手册

- [ORT EP 构建、部署与特性使能](ort-ep-build-and-deploy.md)
- [TensorFlow predictor 构建与部署](tf-build-and-deploy.md)

## 分层

| 层 | 目录 | 内容 | 运行入口 |
|---|---|---|---|
| L0 单元 | `unit/` | C++ GTest，按生产子系统分目录 | `ctest --test-dir build` |
| L1 IR/Pass | `FileCheck/` | ATIR/Conversion/Target 的 `// RUN:` + FileCheck 用例 | `bash tests/FileCheck/run.sh` |
| L2 集成 | `integration/` | 驱动已构建二进制的 pytest；ORT EP 的 CTest Python fixture/e2e | `pytest tests/`、`ctest` |
| L4 端到端 | `e2e/` | 需要外部模型、predictor、安装前缀或人工环境的脚本 | 见各目录说明 |
| 夹具 | `fixtures/` | 跨套件共享的小输入 | 由用例引用 |

```
tests/
├── CMakeLists.txt          # 测试注册的顶层入口（各套件 CMakeLists 由它引入；GTest/Python 发现、coverage）
├── README.md
├── unit/
│   ├── annc/               # op_coverage、JIT cache、template fingerprint、ciface dispatch
│   ├── kernels/            # kernel registry/priority、builtin spec、execution context、GEMM 等
│   ├── onnx2atir/          # ONNX 前端 GTest
│   └── ort_ep/             # ORT EP 编译/执行上下文 GTest
├── FileCheck/              # 52 个 .mlir + 自制 runner（lit 迁移另立任务）
├── integration/
│   ├── cmake/              # CMake 模块契约（KDNN import 能力）pytest
│   ├── fusion/             # config fusion 集成 pytest
│   ├── jit/                # JIT 缓存与预打包权重 pytest
│   ├── onnx2atir/          # annc-onnx2atir CLI 契约 pytest
│   └── ort_ep/             # CTest 驱动的 EP Python fixture/e2e
├── e2e/
│   ├── annc-driver/        # 手工 driver/demo 资产（不接 CTest）
│   ├── build/              # build.sh 选项契约、双框架安装冒烟
│   ├── python/             # Python 绑定手工 smoke（需 --enable-python）
│   └── tf/                 # TF predictor / KP embedding / model zoo 脚本
└── fixtures/
    ├── kp_fused_gather_match.pbtxt
    └── legacy/             # 尚未接线的历史输入
```

## 运行

```bash
# C++ GTest（全部已注册用例）
ctest --test-dir build --output-on-failure

# Python 集成用例；缺可选依赖或构建产物时 skip
python3 -m pytest tests/ -v
# 构建目录不在 build/ 时：
ANNC_BUILD_DIR=/path/to/build python3 -m pytest tests/ -v

# FileCheck
# ANNC_LLVM_BIN 不设置时读取 build/annc_llvm_bin.txt（configure 写入）
PATH="$PWD/build/bin:$ANNC_LLVM_BIN:$PATH" \
LIBRARY_PATH="$PWD/install/lib64:$PWD/build/lib" \
ANNC_GEMM_ASM_SOURCE_DIR=/path/to/GEMM-ASM \
bash tests/FileCheck/run.sh
```

前置条件与 skip 规则：

- `pytest` 必须安装在所选 Python 解释器中；ONNX 前端用例还需要 `onnx` 包。
- `integration/onnx2atir/` 需要 Python `onnx` 包与已构建的 `annc-onnx2atir`；
  缺任一条件时模块整体 skip。
- `integration/fusion/` 需要 `annc-tf2atir`、`annc-opt`、`annc-asm`。
- `integration/jit/` 需要 `numpy`、原生 AArch64 执行环境与已构建的
  `annc`/`annc-asm`；缺任一条件时 skip。
- `integration/cmake/` 需要 `cmake`、`g++` 与 `ar`，缺任一时 skip。
- `integration/ort_ep/` 与 `unit/ort_ep/` 只在
  `ANNC_ENABLE_ONNXRUNTIME_EP=ON` 时注册；Python 驱动还需要 `onnx` 与
  `onnxruntime`。若构建用的解释器没有这两个包，configure 时用
  `-DANNC_TEST_PYTHON=/path/to/venv/bin/python3` 指定测试解释器（该变量只影响
  CTest 的 Python 用例，不会改变 Python 绑定的构建解释器）。
- `unit/annc/annc_jit_cache_test` 只在 `ANNC_ENABLE_TENSORFLOW_ADDON=ON` 时注册；
  ONNX 前端 GTest 只在 `ANNC_ENABLE_ONNX=ON` 时注册。
- `e2e/tf/` 的脚本需要外部 predictor、SavedModel、TSV、brpc client 等，
  由脚本头部的环境变量说明控制；`test_model_zoo_annc.sh` 的默认
  `TFSERVER_PATH` / `MODEL_BASE` 是外部路径，运行前必须覆盖。

## 资产归属（迁移后）

| 测试资产 | 层 | 说明 |
|---|---|---|
| `unit/annc/op_coverage_test.cpp` | L0 | ATIR op 构建覆盖 |
| `unit/annc/jit_template_fingerprint_test.cpp` | L0 | JIT 模板指纹 |
| `unit/annc/annc_jit_cache_test.cpp` | L0 | JIT 编译缓存（TF addon 门控） |
| `unit/annc/ciface_dispatch_test.cpp` | L0 | `mlir_ciface_` 参数派发 |
| `unit/kernels/*_test.cpp`（8 个） | L0 | Kernel registry/priority、builtin spec、execution context、GEMM planner/epilogue/smallshape、masked_mul、customcall |
| `unit/onnx2atir/onnx_frontend_test.cpp` | L0 | ONNX 前端各 stage |
| `unit/ort_ep/annc_ep_compile_test.cpp` | L0 | EP 编译/缓存签名 |
| `unit/ort_ep/annc_ep_execution_test.cpp` | L0 | EP 执行上下文错误记账（fake OrtApi） |
| `FileCheck/**/*.mlir` | L1 | pass/IR 行为 |
| `integration/onnx2atir/test_onnx_frontend.py` | L2 | `annc-onnx2atir` CLI 契约 |
| `integration/cmake/test_kdnn_build.py` | L2 | `cmake/ThirdPartyKDNN.cmake` import 能力探测 |
| `integration/cmake/test_gemm_microkernel_availability.py` | L2 | 微内核归档有/无两种构建下的工具启动、普通 lowering 与预打包能力诊断（用 `ANNC_BUILD_DIR` 选择构建） |
| `integration/fusion/test_config_fusion_kp_fused_gather.py` | L2 | config fusion 集成 |
| `integration/fusion/test_epilogue_source_metadata.py` | L2 | epilogue kernel 克隆的源位置元数据 |
| `integration/jit/test_jit_prepack.py` | L2 | JIT 预打包权重缓存隔离与复用（数值） |
| `integration/ort_ep/*.py` | L2 | EP AOT/context fixture、FTDMM、fusion toggles、context-ABI e2e |
| `e2e/tf/e2e_tf_predictor.py` | L4 | TF predictor + addon 端到端 |
| `e2e/tf/e2e_kp_embedding_patterns.py` | L4 | KP embedding pattern 端到端 |
| `e2e/tf/test_model_zoo_annc.sh` | L4 | model zoo 性能/正确性对拍 |
| `e2e/build/build_mode_selection_test.sh` | L2/build | `build.sh --dry-run` 选项契约 |
| `e2e/build/dual_framework_smoke.sh` | L4/build | 双框架安装前缀冒烟 |
| `e2e/python/matmul_api_smoke.py` | L4/manual | Python 绑定导入与 TensorType 手工 smoke（不被 pytest 收集；需 `--enable-python`） |
| `e2e/annc-driver/` | L4/manual | 历史 driver/demo 资产，未接 CTest |
| `fixtures/legacy/config-fusion-parser-smoke.json` | 夹具 | 暂未被引用的历史 parser-smoke 配置 |

表中未逐一列出的辅助文件（GTest 的 `auto_kernel_*`、stub，FileCheck 的
`Inputs/` JSON，套件内的 README 等）随所属套件目录归类；`git ls-files tests/`
中的每个受跟踪文件都落在上表某个目录或通配模式内。

## 已知基线

2026-10-26（ANNC `2c0aabe`，openEuler 22.03 + 自建 LLVM/MLIR 21.1.3，`build/` 选项
ONNX=ON、ONNXRUNTIME_EP=OFF、TENSORFLOW_ADDON=ON、KDNN=OFF、PYTHON=OFF）实测：

- `ctest -N`：205 项（204 项执行 + 1 项 `Disabled`：
  `MaskedMulContext.Model1FanInMicrobenchmark`）。`ONNXRUNTIME_EP=ON`、`PYTHON=ON`
  的配置为 214 项，差额即这两项开关注册的用例。
- 全量 `ctest`：203/204 通过，1 项失败：`LayerNormContext.SingleColumnMatchesReference`
  实际 `-0.49999618530273438`、期望 `-0.5`，差 3.8e-06 超过 `tolerance*scale = 1e-06`，
  属浮点末位误差，未修改断言。
- `python3 -m pytest tests/`：25 passed + 1 skipped（ONNX 未安装，
  `tests/integration/onnx2atir/test_onnx_frontend.py` 整体 skip）。
- `bash tests/FileCheck/run.sh`：49 PASS / 7 FAIL。其中 6 项为迁移前既有基线：
  `Conversion/atir-to-linalg.mlir`、`Dialect/Atir/config-fusion-constant-operand.mlir`、
  `Dialect/Atir/config-fusion-multi-output.mlir`、`Dialect/Atir/fast-codegen.mlir`、
  `Dialect/Atir/gemm-epilogue-fusion.mlir`、`Dialect/Atir/kp-fused-gather.mlir`；
  第 7 项 `Target/aarch64/annc-aarch64-gemm-pipeline.mlir` 需要
  `ANNC_GEMM_ASM_SOURCE_DIR` 与 `LIBRARY_PATH`，且仓库内 tuning JSON 版本与代码
  要求（`version = 3`）不一致，本机不满足前置条件。

既有失败不要在顺手改动中修改断言或跳过用例。

融合相关用例（`op-fusion.mlir`、`identity-canonicalize-op-fusion.mlir`、
`annc-aarch64-gemm-pipeline.mlir`）显式设置 `ANNC_GEMM=2`：dev 的
`ANNC_GEMM` 门控默认关闭，未设置时 MatMul 不融合。`ANNC_GEMM=2/99` 才会消费
`annc.gemm.epilogue.candidate`；`tests/test_epilogue_source_metadata.py` 已在子进程
环境中注入默认值 `2`。
