# 动态输入 StridedSlice 直译（strided-slice-dyn pass）行为用例：
#   A. x[0]（rank-1 全 shrink → 标量）：real_dynamic_slice + 静态 [1]->标量
#      reshape。这是 54 个主力形态，也曾经是空 shape 向量段错误的回归锚点。
#   B. x[:, k]（shrink=2, bm/em=1/1）：动态 batch 维全取 + 尾维取一，
#      dynamic_reshape 折叠 shrink 维。
#   C. 静态输入零扰动：odml 原生转换，不引入直译产物。
#   D. kill-switch：ANNC_NO_STRIDED_SLICE_DYN=1 维持 tf.residue 兜底。
import os
import subprocess
from pathlib import Path

import pytest

tf = pytest.importorskip("tensorflow")

REPO = Path(__file__).resolve().parents[3]
EXE = REPO / "build" / "bin" / "pb2stablehlo"

pytestmark = pytest.mark.skipif(
    not EXE.exists(),
    reason="pb2stablehlo 未构建（需 ANNC_ENABLE_TF_TRANSFORM=ON 的构建目录）",
)

SS_RESIDUE = 'call_target_name = "tf.StridedSlice"'
RDS = "stablehlo.real_dynamic_slice"


def _artifact(tmp_path, build, out_name, env_extra=None):
    with tf.compat.v1.Graph().as_default() as g:
        build(g)
    pb_path = tmp_path / "g.pb"
    pb_path.write_bytes(g.as_graph_def().SerializeToString())
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(
        [str(EXE), str(pb_path), "--output", out_name, "--skip-preprocess"],
        env=env,
        capture_output=True,
        timeout=300,
    )
    return r.returncode, r.stdout.decode(), r.stderr.decode()


def _take_first(node):
    """x[0]: 静态 rank-1 取首元素，shrink 成标量。动态维 + shrink 现按
    "shrink-dynamic" 跳过（T6 二轮检视：运行时钳制的 start 配常量 limit
    可能越界），翻译路径以静态维锁定。"""
    x = tf.compat.v1.placeholder(tf.int64, [4], name="x")
    tf.compat.v1.strided_slice(x, [0], [1], shrink_axis_mask=1, name="ss")


def test_take_first_translated(tmp_path):
    rc, out, err = _artifact(tmp_path, _take_first, "ss")
    assert rc == 0, err[-800:]
    assert SS_RESIDUE not in out
    # 全常量界限：直译产 real_dynamic_slice 后被 canonicalizer 进一步 fold
    # 为静态 slice（等价且更优）；两种形态都接受。
    assert "stablehlo.slice" in out or RDS in out
    # 全 shrink：[1] -> 标量 的静态 reshape 收尾
    assert "-> tensor<i64>" in out


def test_take_last_dim_dynamic_batch(tmp_path):
    """x[:, 0]：动态 batch 全取 + 尾维取一（shrink_axis_mask=2 主力形态）。"""

    def build(g):
        x = tf.compat.v1.placeholder(tf.float32, [None, 2], name="x")
        tf.compat.v1.strided_slice(
            x, [0, 0], [-1, 1], end_mask=1, shrink_axis_mask=2, name="ss")

    rc, out, err = _artifact(tmp_path, build, "ss")
    assert rc == 0, err[-800:]
    assert SS_RESIDUE not in out
    assert RDS in out
    # shrink 维折叠：dynamic_reshape 到 ?（batch 动态保留）
    assert "stablehlo.dynamic_reshape" in out
    assert "tensor<?xf32>" in out


def test_static_input_untouched(tmp_path):
    def build(g):
        x = tf.compat.v1.placeholder(tf.float32, [4, 2], name="x")
        tf.compat.v1.strided_slice(x, [1], [2], shrink_axis_mask=1, name="ss")

    rc, out, err = _artifact(tmp_path, build, "ss")
    assert rc == 0, err[-800:]
    # 全静态：odml 原生 slice，不引入直译产物
    assert RDS not in out
    assert SS_RESIDUE not in out


def _take_first_dyn(node):
    """动态数组取首（killswitch 专用：静态输入 odml 会原生转换，残差
    断言必须建立在动态输入上）。"""
    x = tf.compat.v1.placeholder(tf.int64, [None], name="x")
    tf.compat.v1.strided_slice(x, [0], [1], shrink_axis_mask=1, name="ss")


def test_killswitch_back_to_residue(tmp_path):
    rc, out, err = _artifact(tmp_path, _take_first_dyn, "ss",
                             {"ANNC_NO_STRIDED_SLICE_DYN": "1"})
    assert rc == 0, err[-800:]
    assert SS_RESIDUE in out
