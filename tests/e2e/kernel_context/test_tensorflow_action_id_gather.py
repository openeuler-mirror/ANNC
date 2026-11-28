"""action_id_gather through TensorFlow fusion, compilation and kernel execution."""

import itertools

import pytest

from tf_graph import build_graph, kernel_spec
from tf_pipeline import run_pipeline


CASES = {
    f"action_r{r1}x{r2}_i{t1}_i{t2}": kernel_spec(
        "kp_embedding_action_id_gather",
        f"KPFusedEmbeddingActionIdGatherR{r1}x{r2}",
        f"float_int{t1}_t_int{t2}_t",
        r1=r1,
        r2=r2,
        t1=t1,
        t2=t2,
    )
    for r1, r2, t1, t2 in itertools.product((1, 2), (1, 2), (32, 64), (32, 64))
}


def _build(case, spec, tf, np, input_, feeds, variants):
    params = input_("params", tf.float32, [None, 2], [[10, 11], [20, 21], [30, 31]])
    v1 = [2, 0] if spec["r1"] == 1 else [[2, 1], [0, 2]]
    v2 = [1, 0] if spec["r2"] == 1 else [[1, 0], [0, 1]]
    i1 = input_("indices1", getattr(tf, f'int{spec["t1"]}'), [None] * spec["r1"], v1)
    i2 = input_("indices2", getattr(tf, f'int{spec["t2"]}'), [None] * spec["r2"], v2)
    rows = input_("pack_dim", tf.int32, [], 2)
    pad = input_("pack", tf.int32, [], 1)
    inner = tf.gather(params, i1, axis=0)
    outer = tf.gather(inner, i2, axis=0)
    reshaped = tf.reshape(outer, tf.stack([rows, -1]))
    output = tf.concat(
        [reshaped, tf.fill(tf.stack([rows, pad]), 0.0)], axis=-1, name="output"
    )
    outputs = [(output, "output")]
    variants.append(("zero_padding", {"pack:0": np.array(0, np.int32)}))
    frontend = [tensor.name for tensor, _ in outputs]
    return outputs, frontend


def make_graph(case):
    return build_graph(case, CASES[case], _build)


@pytest.mark.parametrize("case", sorted(CASES))
def test_production_kernel_full_pipeline(case, tmp_path):
    run_pipeline(CASES[case], make_graph, case, tmp_path)
