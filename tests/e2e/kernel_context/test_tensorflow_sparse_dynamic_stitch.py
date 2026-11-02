"""sparse_dynamic_stitch through TensorFlow fusion, compilation and kernel execution."""

import pytest

from tf_graph import build_graph, kernel_spec
from tf_pipeline import run_pipeline


CASES = {
    f"sparse_dynamic_stitch_n{n}": kernel_spec(
        "kp_sparse_dynamic_stitch", "KPFusedSparseDynamicStitch", n=n
    )
    for n in [*range(2, 18), 36, 37]
}


def _build(case, spec, tf, np, input_, feeds, variants):
    n = spec["n"]
    x = input_("x", tf.int64, [None], [n - 1, n, 0, n - 1])
    tables = [
        input_(
            f"variable_{i}",
            tf.float32,
            [2, 2],
            [[i * 10 + 1, i * 10 + 2], [i * 10 + 3, i * 10 + 4]],
        )
        for i in range(n)
    ]
    partition = tf.cast(tf.math.floormod(x, n), tf.int32)
    positions = tf.dynamic_partition(tf.range(tf.size(x)), partition, n)
    indices = tf.dynamic_partition(tf.math.floordiv(x, n), partition, n)
    values = [tf.gather(table, index, axis=0) for table, index in zip(tables, indices)]
    output = tf.raw_ops.ParallelDynamicStitch(
        indices=positions, data=values, name="output"
    )
    outputs = [(output, "output")]
    variants.append(("all_partitions", {"x:0": np.arange(2 * n, dtype=np.int64)}))
    frontend = [tensor.name for tensor, _ in outputs]
    return outputs, frontend


def make_graph(case):
    return build_graph(case, CASES[case], _build)


@pytest.mark.parametrize("case", sorted(CASES))
def test_production_kernel_full_pipeline(case, tmp_path):
    run_pipeline(CASES[case], make_graph, case, tmp_path)
