# Python 绑定

`python/annc` 使用 pybind11 暴露 ATIR 构建接口（nanobind 为 MLIR 自带 Python 绑定的传递依赖）。

> Python 绑定默认不构建；使用 `./build.sh --enable-python`（或 `-DANNC_ENABLE_PYTHON=ON`）开启。

| 模块 | 说明 |
|------|------|
| `annc.builder` | `MLIRBuilder`，创建 MLIR module/function |
| `annc.helper` | `kp.jit`，从 Python 函数构建 ATIR module |
| `annc.ops` | `matmul`、`add`、`relu`、`constant`、`ret` 等 ATIR Op 构建函数，以及 `ForLoop` 类（实例挂在 `Atir.for_loop`） |
| `annc.types` | `TensorType`、`ElemType`、`Value` |
| `annc.dialects` | ATIR dialect 注册 |

## 示例

用 `kp.jit` 装饰一个 Python 函数即可构建 ATIR module：

```python
import numpy as np

from annc.helper import kp
from annc.ops import matmul
from annc.types import ElemType


@kp.jit
def fused_matmul(inputs, **kwargs):
    lhs = inputs[0]
    rhs = inputs[1]
    cm = inputs[2]
    matmul(ElemType.FP32(), [1024, 1024], lhs, rhs, cm, False)


module = fused_matmul(
    name="demo",
    inputs=[
        np.random.randn(1024, 1024).astype(np.float32),
        np.random.randn(1024, 1024).astype(np.float32),
        np.zeros((1024, 1024)).astype(np.float32),
        np.random.randn(1024).astype(np.float32),
    ],
)

module.operation.print(large_elements_limit=16)
```

## 常见问题

- **导入失败**：确认构建时加了 `--enable-python`、已安装 `pybind11` 与 `nanobind`，且使用了正确的 Python 解释器（`build.sh` 读取 `PYTHON`，并传递给 `Python_EXECUTABLE` / `Python3_EXECUTABLE`）。
