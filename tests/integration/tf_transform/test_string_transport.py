# string 传输包装层（annc.encodeString/decodeString 类型三明治）行为用例：
#   A. string Const -> Reshape：Const 编译期打包 + Reshape retype 为 i8 +
#      官方 legalize 产出 stablehlo.reshape + 恰 1 个译码边界
#   B. ANNC_NO_STRING_TRANSPORT=1 -> 回到 residue（A/B 回归锚点）
#   C. 纯数值 Reshape：零 annc.encodeString（数值路径零扰动）
#   D. Transpose / ConcatV2：官方 legalize 接管的两个代表
#   E. GatherV2（按下标选取族统一进三明治流程）
#   F. 运行时 string 无宽度契约 -> 不包装回 residue（能力门）
#   F2. 零声明动态行宽：feeder string 无需任何参数即可包装
#
# 前置：ANNC_ENABLE_TF_TRANSFORM=ON 构建的 build/bin/pb2stablehlo 与可导入的
# pip TensorFlow；缺则整文件 skip。
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

ENC = 'call_target_name = "annc.encodeString"'
DEC = 'call_target_name = "annc.decodeString"'


def _graphdef(build):
    g = tf.Graph()
    with g.as_default():
        build(g)
    return g.as_graph_def().SerializeToString()


def _artifact(tmp_path, build, out_name, env_extra=None, args=None):
    """运行 pb2stablehlo，返回 (returncode, stablehlo 产物文本, stderr)。"""
    pb = tmp_path / "g.pb"
    pb.write_bytes(_graphdef(build))
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(
        [str(EXE), str(pb), "--output", out_name, "--skip-preprocess"]
        + (args or []),
        env=env,
        capture_output=True,
        timeout=300,
    )
    return r.returncode, r.stdout.decode(), r.stderr.decode()


def _build_const_reshape(g):
    v = tf.constant([["ab", "cd"], ["ef", "gh"]], name="vocab")  # string [2,2]
    tf.reshape(v, [4], name="rs")


def test_const_reshape_sandwich(tmp_path):
    rc, out, err = _artifact(tmp_path, _build_const_reshape, "rs")
    assert rc == 0, err[-800:]
    # Reshape/Const 不再以 residue 载体出现
    assert 'call_target_name = "tf.Reshape"' not in out
    assert 'call_target_name = "tf.Const"' not in out
    # Const 编译期打包 + 官方 legalize 转 stablehlo.reshape + 常量折叠，
    # 终态为单常量：4 行各 [len=2, payload, 0 填充] —— reshape 语义的直接证据
    assert "dense<[[2, 0, 0, 0, 97, 98], [2, 0, 0, 0, 99, 100], [2, 0, 0, 0, 101, 102], [2, 0, 0, 0, 103, 104]]>" in out
    assert "tensor<4x6xi8>" in out
    # 恰 1 个译码边界（Const 折叠不产生编码边界）
    assert out.count(DEC) == 1
    assert ENC not in out
    assert "tf_type.string" in out


def test_killswitch_back_to_residue(tmp_path):
    rc, out, err = _artifact(
        tmp_path, _build_const_reshape, "rs",
        {"ANNC_NO_STRING_TRANSPORT": "1"})
    assert rc == 0, err[-800:]
    assert 'call_target_name = "tf.Reshape"' in out
    assert "annc.encodeString" not in out


def test_numeric_untouched(tmp_path):
    def build(g):
        v = tf.constant([[1.0, 2.0], [3.0, 4.0]], name="x")
        tf.reshape(v, [4], name="rs")

    rc, out, err = _artifact(tmp_path, build, "rs")
    assert rc == 0, err[-800:]
    assert "annc.encodeString" not in out


def test_transpose_sandwich(tmp_path):
    def build(g):
        v = tf.constant([["ab", "cd"], ["ef", "gh"]], name="vocab")
        tf.transpose(v, name="tp")

    rc, out, err = _artifact(tmp_path, build, "tp")
    assert rc == 0, err[-800:]
    assert 'call_target_name = "tf.Transpose"' not in out
    assert "stablehlo.transpose" in out  # 官方 legalize 产出
    assert out.count(DEC) == 1
    assert ENC not in out


def test_concat_sandwich(tmp_path):
    def build(g):
        a = tf.constant([["ab"], ["cd"]], name="a")
        b = tf.constant([["ef"], ["gh"]], name="b")
        tf.concat([a, b], 0, name="cc")

    rc, out, err = _artifact(tmp_path, build, "cc")
    assert rc == 0, err[-800:]
    assert 'call_target_name = "tf.ConcatV2"' not in out
    # 零声明动态行宽：多 Const 汇合不静态折叠，统一动态行宽 + 归一程序
    #（max/subtract/broadcast/concat 全原生）+ 原生 concatenate
    assert "tensor<4x1x?xi8>" in out
    assert "stablehlo.max" in out
    assert "stablehlo.dynamic_broadcast_in_dim" in out
    assert out.count(DEC) == 1
    assert out.count(ENC) == 2


def test_gatherv2_sandwich(tmp_path):
    """按下标选取族：indices 是数值操作数（保持不动），params 走编码。"""

    def build(g):
        p = tf.constant([["ab", "cd"], ["ef", "gh"]], name="params")
        idx = tf.compat.v1.placeholder(tf.int32, [2], name="idx")
        tf.gather(p, idx, name="g")

    rc, out, err = _artifact(
        tmp_path, build, "g", args=["--input", "idx:DT_INT32:2"])
    assert rc == 0, err[-800:]
    assert 'call_target_name = "tf.GatherV2"' not in out
    # 官方 legalize 对 GatherV2(i8) 的 dtype 无关 lowering：
    # 本 wheel 落为 stablehlo.torch_index_select（行级 gather）——实证名单成员
    assert "stablehlo.torch_index_select" in out
    assert "tensor<2x2x6xi8>" in out  # 编码常量与结果贯穿 retype
    assert out.count(DEC) == 1
    assert ENC not in out  # params 为 Const 源，编译期打包


def test_unranked_string_combined_wrap(tmp_path):
    """unranked string + 宽度契约：组合边界链（getDim 治秩 + transform 治
    类型 + setDim 退秩）。cvr_4 ExpandDims 链形态回归锚点（原则一）。
    unranked 经未注册 KP 型算子构造（无形状函数 → 导入后 unranked）。"""
    from tensorflow.core.framework import graph_pb2
    from tensorflow.core.framework import tensor_pb2
    from tensorflow.core.framework import types_pb2

    def build(node):
        s = node.add()
        s.name = "s"
        s.op = "Placeholder"
        s.attr["dtype"].type = types_pb2.DT_STRING
        kp = node.add()
        kp.name = "kp"
        kp.op = "MyHashOp"
        kp.input.append("s:0")
        kp.attr["T"].type = types_pb2.DT_STRING
        dim = node.add()
        dim.name = "dim"
        dim.op = "Const"
        dim.attr["dtype"].type = types_pb2.DT_INT32
        t = tensor_pb2.TensorProto()
        t.dtype = types_pb2.DT_INT32
        t.int_val.append(0)
        dim.attr["value"].tensor.CopyFrom(t)
        e = node.add()
        e.name = "e"
        e.op = "ExpandDims"
        e.input.extend(["kp:0", "dim:0"])
        e.attr["T"].type = types_pb2.DT_STRING  # 供消费方回填 kp 的 string dtype

    g = graph_pb2.GraphDef()
    build(g.node)
    pb = tmp_path / "g.pb"
    pb.write_bytes(g.SerializeToString())
    r = subprocess.run(
        [str(EXE), str(pb), "--output", "e", "--skip-preprocess",
         "--input", "s:DT_STRING"],
        capture_output=True, timeout=300)
    rc, out, err = r.returncode, r.stdout.decode(), r.stderr.decode()
    assert rc == 0, err[-800:]
    # 无 string 残留；组合边界齐备
    assert 'call_target_name = "tf.ExpandDims"' not in out
    assert "annc.getDim" in out            # 治秩入口
    assert out.count(ENC) == 1             # 治类型编码
    assert out.count(DEC) == 1             # 治类型译码
    assert "annc.setDim" in out            # 出口退秩
    assert "tensor<?x?xi8>" in out         # packed 类型贯穿（动态行宽）


def test_runtime_string_dynamic_shape_wraps(tmp_path):
    """动态形状零声明：ranked-动态 string 实例也包装（原则一：
    其他类型域能转的 op，string 域同样能转就转——wd_dcn StridedSlice
    链形态回归锚点）。"""

    def build(g):
        s = tf.compat.v1.placeholder(tf.string, [None, None], name="s")
        c = tf.constant([["ef"], ["gh"]], name="b")
        tf.concat([s, c], 0, name="cc")

    rc, out, err = _artifact(
        tmp_path, build, "cc", args=["--input", "s:DT_STRING"])
    assert rc == 0, err[-800:]
    # 包装发生：无 string 残留；两股数据均动态行宽编码
    assert 'call_target_name = "tf.ConcatV2"' not in out
    assert out.count(ENC) == 2
    assert out.count(DEC) == 1
    assert "tensor<?x?x?xi8>" in out  # 动态维贯穿 retype（含动态行宽）


def test_runtime_string_multi_input_normalize(tmp_path):
    """汇合归一：多股 packed 的运行时行宽各自测量，op2 汇合前发全原生的
    对齐程序（max/subtract/broadcast/concat）。"""

    def build(g):
        s = tf.compat.v1.placeholder(tf.string, [2, 1], name="s")
        c = tf.constant([["ef"], ["gh"]], name="b")
        tf.concat([s, c], 0, name="cc")

    rc, out, err = _artifact(
        tmp_path, build, "cc", args=["--input", "s:DT_STRING"])
    assert rc == 0, err[-800:]
    assert 'call_target_name = "tf.ConcatV2"' not in out
    assert "stablehlo.concatenate" in out  # 官方 legalize 产出
    assert "stablehlo.max" in out          # 汇合归一：行宽取 max
    assert "stablehlo.dynamic_broadcast_in_dim" in out  # 补零零块
    # 两股数据均动态行宽编码 + 1 个译码边界
    assert out.count(ENC) == 2
    assert out.count(DEC) == 1
