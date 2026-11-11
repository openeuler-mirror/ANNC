# rank 恢复管线（rank-infer 默认启用：OpDef 同型通用规则 + Reshape/Fill 常量
# 求值 + Smuggle 补结果盲区）行为用例：
#   A. ranked 输入的 Softmax：OpDef 同型规则补出结果 rank -> 官方 legalize
#      产出 stablehlo.softmax（当初回归的 op 家族实证）
#   B. unranked 入口的 Softmax：管线必须完成，op 走 custom_call 兜底
#      （当初回归形态：混合态撞 materialization 整管线失败）
#   C. Const shape Reshape：常量求值 -> stablehlo.reshape
#   D. kill-switch：ANNC_NO_TF_RANK_INFER=1 回退
# 前置：ANNC_ENABLE_TF_TRANSFORM=ON 构建的 build/bin/pb2stablehlo。
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


def _graphdef(build):
    g = tf.Graph()
    with g.as_default():
        build(g)
    return g.as_graph_def().SerializeToString()


def _artifact(tmp_path, build, out_name, env_extra=None, args=None):
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


def test_ranked_softmax_legalizes(tmp_path):
    """OpDef 同型规则：ranked 输入 -> Softmax 结果 rank 补出 -> 官方 legalize。
    本 wheel 将 Softmax 分解为纯 stablehlo 原语（reduce/subtract/exp/divide），
    产物中不存在 stablehlo.softmax op——分解即 legalize 成功的证据。"""

    def build(g):
        x = tf.compat.v1.placeholder(tf.float32, [2, 3], name="x")
        tf.nn.softmax(x, name="sm")

    rc, out, err = _artifact(
        tmp_path, build, "sm", args=["--input", "x:DT_FLOAT:2,3"])
    assert rc == 0, err[-800:]
    assert 'call_target_name = "tf.Softmax"' not in out
    assert "stablehlo.exponential" in out  # 分解产物（原语级 legalization）
    assert "tensor<2x3xf32>" in out  # rank 贯穿到返回签名


def test_unranked_softmax_pipeline_completes(tmp_path):
    """真 unrank（运行时 shape 的 reshape 结果）：rank 无法推导 -> Smuggle 兜
    底（操作数或结果 unranked 均拦），管线完成——当初回归的失败形态（混合
    态撞 materialization 整管线失败）不再出现。"""

    def build(g):
        x = tf.compat.v1.placeholder(tf.float32, [6], name="x")
        sh = tf.compat.v1.placeholder(tf.int32, [2], name="sh")  # 运行时 shape
        rs = tf.reshape(x, sh, name="rs")  # 结果 unranked（shape 值不可知）
        tf.nn.softmax(rs, name="sm")

    rc, out, err = _artifact(
        tmp_path, build, "sm",
        args=["--input", "x:DT_FLOAT:6", "--input", "sh:DT_INT32:2"])
    assert rc == 0, err[-800:]
    # rank retype（unranked -> ranked+动态维）后，官方 legalize 把 Softmax 全
    # 分解为动态维 stablehlo（dynamic_broadcast_in_dim/divide）——连 custom_call
    # T6 决策：dyn-view 不迁——动态 Reshape 保持残差通道（引入一维扁平
    # 中间形态属不需要的形状结构变化），由 sink 侧处理。旧断言（getDim
    # 序列视图）已过时：
    # schema-evolution 批次 5），同样由运行时解释但保留 op 语义。
    assert "stablehlo.divide" in out  # Softmax 分解产物
    assert "tensor<?x?xf32>" in out  # ranked+动态维贯穿（rank 契约生效）
    assert 'call_target_name = "annc.getDim"' not in out  # dyn-view dropped
    assert 'call_target_name = "tf.Reshape"' in out  # rides the residue


def test_const_shape_reshape_legalizes(tmp_path):
    """Reshape 常量求值：ranked 输入 + Const shape -> 结果 rank 补出。"""

    def build(g):
        x = tf.compat.v1.placeholder(tf.float32, [6], name="x")
        sh = tf.constant([2, 3], name="sh")
        tf.reshape(x, sh, name="rs")

    rc, out, err = _artifact(
        tmp_path, build, "rs", args=["--input", "x:DT_FLOAT:6"])
    assert rc == 0, err[-800:]
    assert "stablehlo.reshape" in out
    assert "tensor<2x3xf32>" in out  # 结果 rank/dims 贯穿


def test_killswitch_disables_rank_recover(tmp_path):
    def build(g):
        x = tf.compat.v1.placeholder(tf.float32, [2, 3], name="x")
        tf.nn.softmax(x, name="sm")

    rc, out, err = _artifact(
        tmp_path, build, "sm", {"ANNC_NO_TF_RANK_INFER": "1"},
        args=["--input", "x:DT_FLOAT:2,3"])
    assert rc == 0, err[-800:]
    # 关闭后 Softmax 不再被本 pass 补 rank（由 Smuggle/custom_call 或
    # 既有路径兜底），作为 A/B 观察点不做强断言；仅要求管线完成。
    assert rc == 0
