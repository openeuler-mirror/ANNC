# unrank 包装（annc.getDim/setDim 序列视图）行为用例：
#   A. 未注册 KP 型算子（unranked 输出）→ Cast：包装后官方 legalize 原生转换，
#      getDim/setDim 边界各 1 个
#   B. kill-switch：ANNC_NO_UNRANK_WRAP=1 回退 unranked custom_call 兜底
#   C. 形状相关族边界：StridedSlice 不包装（序列域不等价），维持兜底
#   D. 已 ranked 实例零扰动：无 annc.getDim/setDim
#   E. 逐元素二元 + 标量：广播标量走 [1] 序列视图
#
# 前置：ANNC_ENABLE_TF_TRANSFORM=ON 构建的 build/bin/pb2stablehlo；缺则整文件
# skip。未注册算子以手工 GraphDef 构造（复现 cvr/adx 的 KP 边界形态——无形状
# 函数 → 导入后 unranked）。
import os
import subprocess
from pathlib import Path

import pytest

tf = pytest.importorskip("tensorflow")
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2

REPO = Path(__file__).resolve().parents[3]
EXE = REPO / "build" / "bin" / "pb2stablehlo"

pytestmark = pytest.mark.skipif(
    not EXE.exists(),
    reason="pb2stablehlo 未构建（需 ANNC_ENABLE_TF_TRANSFORM=ON 的构建目录）",
)

GETDIM = 'call_target_name = "annc.getDim"'  # tf.residue 载体属性形态
SETDIM = 'call_target_name = "annc.setDim"'
F32 = types_pb2.DT_FLOAT
_TYPE_ATTRS = {"T", "SrcT", "DstT", "dtype"}  # dtype 类属性落在 .type 字段


def _set_attr(node, key, value):
    if key in _TYPE_ATTRS:
        node.attr[key].type = int(value)  # DataType 枚举值
    elif isinstance(value, bool):
        node.attr[key].b = int(value)
    elif isinstance(value, int):
        node.attr[key].i = value
    else:
        node.attr[key].tensor.CopyFrom(value)


def _graphdef(build):
    g = graph_pb2.GraphDef()  # 未注册算子无法经 tf 构图 API 表达，直接搭 proto
    build(g.node)
    return g.SerializeToString()


def _artifact(tmp_path, build, out_name, env_extra=None):
    pb = tmp_path / "g.pb"
    pb.write_bytes(_graphdef(build))
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(
        [str(EXE), str(pb), "--output", out_name, "--skip-preprocess"],
        env=env,
        capture_output=True,
        timeout=300,
    )
    return r.returncode, r.stdout.decode(), r.stderr.decode()

def _build_kp_cast(node):
    """未注册 KP 型算子（无形状函数 → 输出 unranked）→ Cast。"""
    x = node.add()
    x.name = "x"
    x.op = "Placeholder"
    _set_attr(x, "dtype", F32)
    kp = node.add()
    kp.name = "kp"
    kp.op = "MyHashOp"
    kp.input.append("x:0")
    _set_attr(kp, "T", F32)
    c = node.add()
    c.name = "c"
    c.op = "Cast"
    c.input.append("kp:0")
    _set_attr(c, "SrcT", F32)
    _set_attr(c, "DstT", types_pb2.DT_INT32)  # 非恒等 cast，确保节点存活
    _set_attr(c, "Truncate", False)


def test_kp_cast_wrapped(tmp_path):
    rc, out, err = _artifact(tmp_path, _build_kp_cast, "c")
    assert rc == 0, err[-800:]
    # Cast 不再以 unranked custom_call 兜底（官方 legalize 原生转换或折叠）
    assert 'call_target_name = "tf.Cast"' not in out
    # 序列视图边界各 1 个
    assert out.count(GETDIM) == 1
    assert out.count(SETDIM) == 1
    # 译码出口保持 unranked 契约（DstT=i32）
    assert "tensor<*xi32>" in out


def test_kp_zeroslike_wrapped(tmp_path):
    """ZerosLike 曾漏在名单外（adx_1 实测残留 3 例）——回归锚点。"""

    def build(node):
        x = node.add()
        x.name = "x"
        x.op = "Placeholder"
        _set_attr(x, "dtype", F32)
        kp = node.add()
        kp.name = "kp"
        kp.op = "MyHashOp"
        kp.input.append("x:0")
        _set_attr(kp, "T", F32)
        z = node.add()
        z.name = "z"
        z.op = "ZerosLike"
        z.input.append("kp:0")
        _set_attr(z, "T", F32)

    rc, out, err = _artifact(tmp_path, build, "z")
    assert rc == 0, err[-800:]
    assert 'call_target_name = "tf.ZerosLike"' not in out
    assert out.count(GETDIM) == 1
    assert out.count(SETDIM) == 1


def test_killswitch_back_to_unranked_fallback(tmp_path):
    rc, out, err = _artifact(
        tmp_path, _build_kp_cast, "c", {"ANNC_NO_UNRANK_WRAP": "1"})
    assert rc == 0, err[-800:]
    assert 'residue_reason = "unranked"' in out
    assert "annc.getDim" not in out


def test_shape_aware_boundary_not_wrapped(tmp_path):
    """形状相关族（Softmax 按轴归一化）：序列域不等价 → 不包装（边界锚点）。
    且无辅助操作数，排除无关干扰。"""

    def build(node):
        x = node.add()
        x.name = "x"
        x.op = "Placeholder"
        _set_attr(x, "dtype", F32)
        kp = node.add()
        kp.name = "kp"
        kp.op = "MyHashOp"
        kp.input.append("x:0")
        _set_attr(kp, "T", F32)
        sm = node.add()
        sm.name = "sm"
        sm.op = "Softmax"
        sm.input.append("kp:0")
        _set_attr(sm, "T", F32)

    rc, out, err = _artifact(tmp_path, build, "sm")
    assert rc == 0, err[-800:]
    assert "@tf.Softmax" in out  # 维持兜底
    assert "annc.getDim" not in out


def test_ranked_untouched(tmp_path):
    """已 ranked 实例零扰动：不需要序列视图。"""

    def build(node):
        x = node.add()
        x.name = "x"
        x.op = "Placeholder"
        _set_attr(x, "dtype", F32)
        x.attr["shape"].shape.dim.add().size = 2
        x.attr["shape"].shape.dim.add().size = 3
        ab = node.add()
        ab.name = "ab"
        ab.op = "Abs"
        ab.input.append("x:0")
        _set_attr(ab, "T", F32)

    rc, out, err = _artifact(tmp_path, build, "ab")
    assert rc == 0, err[-800:]
    assert "annc.getDim" not in out
    assert "annc.setDim" not in out


def _build_kp_binary_scalar(node):
    """unranked KP 一元 + 标量占位喂 KP 二元：T6 决策后二元不压平。"""
    x = node.add()
    x.name = "x"
    x.op = "Placeholder"
    _set_attr(x, "dtype", F32)
    kp = node.add()
    kp.name = "un"
    kp.op = "MyHashOp"
    kp.input.append("x:0")
    _set_attr(kp, "T", F32)
    sc = node.add()
    sc.name = "sc"
    sc.op = "Placeholder"
    _set_attr(sc, "dtype", F32)
    bi = node.add()
    bi.name = "y"
    bi.op = "MyBinaryOp"
    bi.input.append("un:0")
    bi.input.append("sc:0")
    _set_attr(bi, "T", F32)


def test_elementwise_binary_not_wrapped(tmp_path):
    """T6 决策：广播二元族不再压平（错配形状会变成错误数学）。unranked
    二元走 Smuggle 兜底，任何形态都不出现 annc.getDim。"""
    pb = tmp_path / "g.pb"
    pb.write_bytes(_graphdef(_build_kp_binary_scalar))
    r = subprocess.run(
        [str(EXE), str(pb), "--output", "y", "--skip-preprocess",
         "--assume-opaque-i8"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert GETDIM not in r.stdout
