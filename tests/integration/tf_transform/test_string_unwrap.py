# string 反包装（string-unwrap）行为用例：
#   A1. 成功三明治保留：legalize 成功的站点（ConcatV2）边界与原生 op 原样
#   A2. 失败三明治完整回退：legalize 失败的站点（StridedSlice）回到从未被
#       包装过的形态——无 encodeString/decodeString/无 i8 packed
#   A3. 站点独立性：失败站点与成功站点同图相邻，回退前者、后者原样
#   A5. 对照等价：全失败图上，默认（包装+回退）产物与
#       ANNC_NO_STRING_TRANSPORT=1（从未包装）产物逐字节一致——反包装无损
#       的最强验收
#   B1. kill-switch：ANNC_NO_STRING_UNWRAP=1 保留失败三明治的 i8 形态（A/B）
#
# 前置：ANNC_ENABLE_TF_TRANSFORM=ON 构建的 build/bin/pb2stablehlo；缺则整
# 文件 skip。失败站点选 StridedSlice（实测本轮 odml 无其 pattern，数值域同
# 样缺失——legalize_tf.cc:3671 要求输入输出全静态，动态行宽下不满足）。
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


def _artifact(tmp_path, build, out_name, env_extra=None):
    pb = tmp_path / "g.pb"
    pb.write_bytes(_graphdef(build))
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(
        [str(EXE), str(pb), "--output", out_name, "--skip-preprocess"]
        + (["--input", "s:DT_STRING"] if env_extra and env_extra.get("FEED") else []),
        env=env,
        capture_output=True,
        timeout=300,
    )
    return r.returncode, r.stdout.decode(), r.stderr.decode()


def _build_stridedslice_only(g):
    """失败站点单例：feeder string → StridedSlice（动态行宽 i8 实例必失败）。"""
    s = tf.compat.v1.placeholder(tf.string, [2, 1], name="s")
    tf.raw_ops.StridedSlice(
        input=s, begin=[0, 0], end=[2, 1], strides=[1, 1], name="sl")


def _build_mixed(g):
    """失败站点与成功站点同图相邻（共用 feeder）：独立性验收。"""
    s = tf.compat.v1.placeholder(tf.string, [2, 1], name="s")
    tf.raw_ops.StridedSlice(
        input=s, begin=[0, 0], end=[2, 1], strides=[1, 1], name="sl")
    b = tf.constant([["ef"], ["gh"]], name="b")
    tf.concat([s, b], 0, name="cc")


def test_success_sandwich_kept(tmp_path):
    """A1：legalize 成功的站点三明治原样保留。"""

    def build(g):
        s = tf.compat.v1.placeholder(tf.string, [2, 1], name="s")
        b = tf.constant([["ef"], ["gh"]], name="b")
        tf.concat([s, b], 0, name="cc")

    rc, out, err = _artifact(tmp_path, build, "cc")
    assert rc == 0, err[-800:]
    assert ENC in out and DEC in out
    assert "stablehlo.concatenate" in out
    assert 'call_target_name = "tf.ConcatV2"' not in out


def test_failed_sandwich_fully_rolled_back(tmp_path):
    """A2：失败站点完整回退——无边界、无 i8、无归一程序，恢复 string 形态。"""

    def build(g):
        s = tf.compat.v1.placeholder(tf.string, [2, 1], name="s")
        tf.raw_ops.StridedSlice(
            input=s, begin=[0, 0], end=[2, 1], strides=[1, 1], name="sl")

    rc, out, err = _artifact(
        tmp_path, build, "sl", env_extra={"FEED": "1"})
    assert rc == 0, err[-800:]
    # 回退四项全负
    assert ENC not in out
    assert DEC not in out
    assert "xi8" not in out
    assert "get_dimension_size" not in out
    # StridedSlice 恢复 string 形态
    assert 'call_target_name = "tf.StridedSlice"' in out
    assert "tf_type.string" in out


def test_sandwich_independence(tmp_path):
    """A3：失败站点与成功站点同图相邻——回退前者，后者原样保留。"""

    rc, out, err = _artifact(tmp_path, _build_mixed, "cc",
                             env_extra={"FEED": "1"})
    assert rc == 0, err[-800:]
    # 成功站点：ConcatV2 三明治原样
    assert ENC in out and DEC in out
    assert "stablehlo.concatenate" in out
    # 失败站点：StridedSlice 回退到 string 形态
    assert "tf_type.string" in out
    sl_i8 = [l for l in out.splitlines()
             if "tf.StridedSlice" in l and "xi8" in l]
    assert not sl_i8  # 失败站点不留 i8 形态


def test_unwrap_equivalent_to_never_wrapped(tmp_path):
    """A5（对照等价，最强验收）：全失败图上，默认（包装+回退）与
    从未包装（ANNC_NO_STRING_TRANSPORT=1）产物逐字节一致——反包装无损。"""

    def build(g):
        s = tf.compat.v1.placeholder(tf.string, [2, 1], name="s")
        tf.raw_ops.StridedSlice(
            input=s, begin=[0, 0], end=[2, 1], strides=[1, 1], name="sl")

    pb = tmp_path / "g.pb"
    pb.write_bytes(_graphdef(build))

    outs = {}
    for tag, env in (("default", None),
                     ("never", {"ANNC_NO_STRING_TRANSPORT": "1"})):
        env2 = dict(os.environ)
        if env:
            env2.update(env)
        r = subprocess.run(
            [str(EXE), str(pb), "--output", "sl", "--skip-preprocess",
             "--input", "s:DT_STRING"],
            env=env2, capture_output=True, timeout=300)
        assert r.returncode == 0
        outs[tag] = r.stdout

    # 允许唯一差异：residue_reason 记录兜底发生的阶段（no_legalize_pattern
    # = legalize 阶段兜底 vs non_numeric_type = restore 阶段兜底），语义等价。
    import re as _re
    norm = lambda b: _re.sub(rb'residue_reason = "[a-z_]+"',
                             b'residue_reason = "<fallback>"', b)
    assert norm(outs["default"]) == norm(outs["never"]), (
        "回退后的产物应与从未包装的产物一致（除兜底阶段记录外逐字节一致——反包装无损）")


def test_unrank_failed_sandwich_rolled_back(tmp_path):
    """A4 组合链回退：unranked string 的 GatherNd（动态行宽 i8 实例实测
    不可转）——unrank 视图（getDim）与 string 边界（encodeString）一并回
    退，恢复 unranked string 原形。unranked 经未注册 KP 型算子构造。"""
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
        idx = node.add()
        idx.name = "idx"
        idx.op = "Const"
        idx.attr["dtype"].type = types_pb2.DT_INT64
        t = tensor_pb2.TensorProto()
        t.dtype = types_pb2.DT_INT64
        t.tensor_shape.dim.add().size = 2
        t.int64_val.extend([0, 1])
        idx.attr["value"].tensor.CopyFrom(t)
        g = node.add()
        g.name = "g"
        g.op = "GatherNd"
        g.input.extend(["kp:0", "idx:0"])
        g.attr["Tparams"].type = types_pb2.DT_STRING
        g.attr["Tindices"].type = types_pb2.DT_INT64

    g = graph_pb2.GraphDef()
    build(g.node)
    pb = tmp_path / "g.pb"
    pb.write_bytes(g.SerializeToString())
    r = subprocess.run(
        [str(EXE), str(pb), "--output", "g", "--skip-preprocess",
         "--input", "s:DT_STRING"],
        capture_output=True, timeout=300)
    rc, out, err = r.returncode, r.stdout.decode(), r.stderr.decode()
    assert rc == 0, err[-800:]
    # GatherNd(i8) 原生化成功（stablehlo.gather）→ 三明治保留：
    # 组合链 getDim 治秩 + encodeString 治类型 + 原生 gather + decodeString
    assert "annc.getDim" in out
    assert ENC in out and DEC in out
    assert "stablehlo.gather" in out
    assert 'call_target_name = "tf.GatherNd"' not in out


def test_pure_unrank_failed_sandwich_rolled_back(tmp_path):
    """纯 unrank 回退（无 string 域）：unranked i64 的 GuaranteeConst（odml
    无 pattern，必然 tf 域残留）——getDim 视图与 setDim 齐撤，GuaranteeConst
    保留并恢复 unranked 契约。此路径曾把 op2 本身误删（悬空），此用例锚定。"""
    from tensorflow.core.framework import graph_pb2
    from tensorflow.core.framework import types_pb2

    g = graph_pb2.GraphDef()
    s = g.node.add()
    s.name = "s"
    s.op = "Placeholder"
    s.attr["dtype"].type = types_pb2.DT_INT64
    kp = g.node.add()
    kp.name = "kp"
    kp.op = "MyHashOp"
    kp.input.append("s:0")
    kp.attr["T"].type = types_pb2.DT_INT64
    gc = g.node.add()
    gc.name = "gc"
    gc.op = "GuaranteeConst"
    gc.input.append("kp:0")
    gc.attr["T"].type = types_pb2.DT_INT64

    pb = tmp_path / "g.pb"
    pb.write_bytes(g.SerializeToString())
    r = subprocess.run(
        [str(EXE), str(pb), "--output", "gc", "--skip-preprocess",
         "--input", "s:DT_INT64"],
        capture_output=True, timeout=300)
    rc, out, err = r.returncode, r.stdout.decode(), r.stderr.decode()
    assert rc == 0, err[-800:]
    # 纯 unrank 三明治完整回退：getDim/setDim 全清
    assert "annc.getDim" not in out
    assert "annc.setDim" not in out
    # GuaranteeConst 保留（tf 域残留形态），输出恢复 unranked
    assert "GuaranteeConst" in out
    assert "tensor<*xi64>" in out


def test_kill_switch_keeps_i8_form(tmp_path):
    """B1：ANNC_NO_STRING_UNWRAP=1 → 失败三明治保留 i8 形态（A/B 对照）。"""
    pb = tmp_path / "g.pb"
    pb.write_bytes(_graphdef(_build_stridedslice_only))

    env = dict(os.environ)
    env.update({"ANNC_NO_STRING_UNWRAP": "1"})
    r = subprocess.run(
        [str(EXE), str(pb), "--output", "sl", "--skip-preprocess",
         "--input", "s:DT_STRING"],
        env=env, capture_output=True, timeout=300)
    assert r.returncode == 0
    out = r.stdout.decode()
    # 失败三明治的 i8 形态保留（未回退）
    assert ENC in out and DEC in out
    assert "?xi8" in out
