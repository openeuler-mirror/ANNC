"""Run the preload plugin against the TF1.15 Python runtime."""

import ctypes
from pathlib import Path
import sys

from google.protobuf import text_format
import numpy as np
import tensorflow as tf


def main(optimizer, fused_op, graph_file):
    sdk = Path(tf.sysconfig.get_lib())
    for library in (
        "libtensorflow_framework.so.1",
        "python/_pywrap_tensorflow_internal.so",
    ):
        ctypes.CDLL(str(sdk / library), mode=ctypes.RTLD_GLOBAL)
    tf.load_library(optimizer)
    tf.load_op_library(fused_op)
    definition = text_format.Parse(Path(graph_file).read_text(), tf.GraphDef())
    with tf.Graph().as_default(), tf.Session() as session:
        tf.import_graph_def(definition, name="")
        actual, expected = session.run(["fused:0", "reference:0"])
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    main(*sys.argv[1:])
