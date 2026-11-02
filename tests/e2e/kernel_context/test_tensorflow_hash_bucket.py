"""hash_bucket through TensorFlow fusion, compilation and kernel execution."""

import pytest

from tf_graph import build_graph, kernel_spec
from tf_pipeline import run_pipeline


CASES = {
    "hash_bucket": kernel_spec(
        "dnn_embedding_hash_bucket", "KPFusedDnnEmbeddingWithHashBucket"
    ),
}


def _build(case, spec, tf, np, input_, feeds, variants):
    strings = input_("input", tf.string, [None], [b"a", b"", b"b"])
    weight = input_(
        "embedding_weight",
        tf.float32,
        [7, 2],
        np.arange(14, dtype=np.float32).reshape(7, 2),
    )
    with tf.name_scope("test_embedding"):
        expanded = tf.expand_dims(strings, -1, name="ExpandDims")
        shape = tf.raw_ops.Shape(input=expanded)
        locations = tf.where(tf.not_equal(expanded, ""))
        values = tf.gather_nd(expanded, locations)
        hashes = tf.strings.to_hash_bucket_fast(values, 7)
        unique, inverse = tf.unique(hashes)
        embeddings = tf.gather(weight, unique, axis=0)
        row_ids = tf.cast(locations[:, 0], tf.int32)
        mean = tf.raw_ops.SparseSegmentMean(
            data=embeddings,
            indices=inverse,
            segment_ids=row_ids,
            name="embedding_lookup_sparse/mean",
        )
        result = tf.reshape(mean, tf.stack([shape[0], 2]), name="Reshape")
        outputs = [(result, "output")]
    variants.append(
        ("unicode_and_repeat", {"input:0": np.array(["你好", "a", "a"], object)})
    )
    frontend = [tensor.name for tensor, _ in outputs]
    return outputs, frontend


def make_graph(case):
    return build_graph(case, CASES[case], _build)


@pytest.mark.parametrize("case", sorted(CASES))
def test_production_kernel_full_pipeline(case, tmp_path):
    run_pipeline(CASES[case], make_graph, case, tmp_path)
