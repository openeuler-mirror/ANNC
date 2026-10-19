# ANNC 测试目录

本目录是 ANNC 所有受跟踪测试的唯一归属地。生产目录（`annc/`、`frontends/`、
`addons/`、`tensorflow_addons/`）只放实现，不得包含测试源码、GTest/Python
测试发现、`gtest_discover_tests` 或 `add_test` 注册；测试到生产目录只允许
单向依赖。

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
│   ├── fusion/             # config fusion 集成 pytest
│   ├── onnx2atir/          # annc-onnx2atir CLI 契约 pytest
│   └── ort_ep/             # CTest 驱动的 EP Python fixture/e2e
├── e2e/
│   ├── annc-driver/        # 手工 driver/demo 资产（不接 CTest）
│   ├── build/              # build.sh 选项契约、双框架安装冒烟
│   ├── python/             # Python 绑定手工 smoke
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
PATH="$PWD/build/bin:$PWD/build/_deps/llvm-build/bin:$PATH" \
LIBRARY_PATH="$PWD/install/lib:$PWD/build/lib" \
ANNC_GEMM_ASM_SOURCE_DIR=/path/to/GEMM-ASM \
bash tests/FileCheck/run.sh
```

前置条件与 skip 规则：

- `pytest` 必须安装在所选 Python 解释器中；ONNX 前端用例还需要 `onnx` 包。
- `integration/onnx2atir/` 需要 Python `onnx` 包与已构建的 `annc-onnx2atir`；
  缺任一条件时模块整体 skip。
- `integration/fusion/` 需要 `annc-tf2atir`、`annc-opt`、`annc-asm`。
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
| `integration/fusion/test_config_fusion_kp_fused_gather.py` | L2 | config fusion 集成 |
| `integration/ort_ep/*.py` | L2 | EP AOT/context fixture、FTDMM、fusion toggles、context-ABI e2e |
| `e2e/tf/e2e_tf_predictor.py` | L4 | TF predictor + addon 端到端 |
| `e2e/tf/e2e_kp_embedding_patterns.py` | L4 | KP embedding pattern 端到端 |
| `e2e/tf/test_model_zoo_annc.sh` | L4 | model zoo 性能/正确性对拍 |
| `e2e/build/build_mode_selection_test.sh` | L2/build | `build.sh --dry-run` 选项契约 |
| `e2e/build/dual_framework_smoke.sh` | L4/build | 双框架安装前缀冒烟 |
| `e2e/python/matmul_api_smoke.py` | L4/manual | Python 绑定导入与 TensorType 手工 smoke（不被 pytest 收集） |
| `e2e/annc-driver/` | L4/manual | 历史 driver/demo 资产，未接 CTest |
| `fixtures/legacy/config-fusion-parser-smoke.json` | 夹具 | 暂未被引用的历史 parser-smoke 配置 |

表中未逐一列出的辅助文件（GTest 的 `auto_kernel_*`、stub，FileCheck 的
`Inputs/` JSON，套件内的 README 等）随所属套件目录归类；`git ls-files tests/`
中的每个受跟踪文件都落在上表某个目录或通配模式内。

## 已知基线

2026-10-18（ANNC `7bff5dc`，`build/` 选项 ONNX=ON、ONNXRUNTIME_EP=ON、
TENSORFLOW_ADDON=ON、KDNN=OFF）实测：

- `ctest -N`：214 项（213 项执行 + 1 项 `Disabled`）。
  - 重构前本机 `ctest -N` 为 215，是因为未跟踪的
    `logs/dffm/tf2atir_tf_frontend_test.cpp` 触发了 `annc-tf2atir` 的日志目录
    死接线，额外注册了 `annc-tf2atir-frontend-test`；删除死接线后为 214。
    干净克隆本来就只有 214。
- 全量 `ctest`（配合 `-DANNC_TEST_PYTHON=<venv>/bin/python3`）：213/213 通过，
  1 项 Disabled。系统 Python 缺 `onnx`/`onnxruntime` 时 EP Python 用例会直接
  `ModuleNotFoundError`，这不是测试逻辑失败。
- `python3 -m pytest tests/`：1 passed + 1 skipped（ONNX 未安装，前端测试模块
  整体 skip）。
- `bash tests/FileCheck/run.sh`：46 PASS / 6 FAIL。失败项为
  `Conversion/atir-to-linalg.mlir`、`Dialect/Atir/config-fusion-constant-operand.mlir`、
  `Dialect/Atir/config-fusion-multi-output.mlir`、`Dialect/Atir/fast-codegen.mlir`、
  `Dialect/Atir/gemm-epilogue-fusion.mlir`、`Dialect/Atir/kp-fused-gather.mlir`。

这些失败是本任务迁移前已存在的基线，不属于本次目录重构的回归；不要在迁移中
顺手修改断言或跳过用例。

融合相关用例（`op-fusion.mlir`、`identity-canonicalize-op-fusion.mlir`、
`annc-aarch64-gemm-pipeline.mlir`）显式设置 `ANNC_GEMM=2`：dev 的
`ANNC_GEMM` 门控默认关闭，未设置时 MatMul 不融合。
