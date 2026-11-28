"""Graph construction shared by per-kernel TensorFlow E2E cases."""


def kernel_spec(
    pattern, op, suffix="float", mode="aot", abi="annc_context_v3", **parameters
):
    symbol = (
        f"ANNCKernel_ContextV3_{op}_{suffix}_auto" if abi == "annc_context_v3" else None
    )
    return dict(pattern=pattern, op=op, mode=mode, abi=abi, symbol=symbol, **parameters)


def scalar_slice(tf, value, begin, name):
    return tf.raw_ops.StridedSlice(
        input=value, begin=begin, end=[1], strides=[1], shrink_axis_mask=1, name=name
    )


def column(tf, keys, begin, name):
    return tf.raw_ops.StridedSlice(
        input=keys,
        begin=begin,
        end=[0, 2],
        strides=[1, 1],
        begin_mask=1,
        end_mask=1,
        shrink_axis_mask=2,
        name=name,
    )


def build_graph(case, spec, builder):
    """Return GraphDef, frontend fetches, fused outputs and feed scenarios."""
    import numpy as np
    import tensorflow as tf

    graph = tf.Graph()
    feeds = {}
    variants = []

    def input_(name, dtype, shape, value):
        tensor = tf.compat.v1.placeholder(dtype, shape, name=name)
        feeds[tensor.name] = np.asarray(value, dtype=dtype.as_numpy_dtype)
        return tensor

    with graph.as_default():
        outputs, frontend = builder(case, spec, tf, np, input_, feeds, variants)
        definition = graph.as_graph_def(add_shapes=True)
    scenarios = [("normal", feeds)]
    scenarios += [(label, {**feeds, **changes}) for label, changes in variants]
    scenarios.append(("repeat", feeds))
    return (
        definition,
        frontend,
        [(tensor.name, slot) for tensor, slot in outputs],
        scenarios,
    )


def reference_graph(make_graph, spec, original):
    case = spec.get("tf1_reference_case")
    if case is None:
        return original

    import numpy as np
    import tensorflow as tf

    if not tf.__version__.startswith("1."):
        return original

    reference = make_graph(case)
    _, _, outputs, scenarios = original
    _, _, reference_outputs, reference_scenarios = reference
    assert outputs == reference_outputs
    assert len(scenarios) == len(reference_scenarios)
    for (label, feeds), (reference_label, reference_feeds) in zip(
        scenarios, reference_scenarios
    ):
        assert label == reference_label
        assert feeds.keys() == reference_feeds.keys()
        for name, values in feeds.items():
            np.testing.assert_array_equal(values, reference_feeds[name])
    return reference
