import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0' 
import tensorflow as tf
import numpy as np
import unittest

class TestSparseSegmentMeanSlice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize test data and custom op"""
        # Load custom op
        cls.custom_op = tf.load_op_library('kp_fused_sparse_dynamic_stitch.so')
        
        cls.variables = []
        max_val = float('inf')
        for i in range(12):
            N_i = np.random.randint(1000000, 44739244)
            max_val = min(N_i, max_val)
            var = tf.Variable(
                tf.random.normal([N_i, 10], dtype=tf.float32),  # shape: (N_i, 10)
                name=f"embedding_table_{i}"
            )
            cls.variables.append(var)
            print(f"Created variable {i}: shape={var.shape}")

        x_np = np.random.randint(0, 12*max_val, size=(10000, 12))
        cls.x = tf.constant(x_np, dtype=tf.int64)
        
        # Create tf session
        cls.sess = tf.compat.v1.Session()
        cls.sess.run(tf.compat.v1.global_variables_initializer())

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()

    def test_base(self):
        x_first = self.sess.run(self.x)
        var_first = self.sess.run(self.variables[0])

        x_second = self.sess.run(self.x)
        var_second = self.sess.run(self.variables[0])
        np.testing.assert_allclose(
            x_first,
            x_second,
            rtol=1e-6,
            err_msg="Input values mismatch"
        )

        np.testing.assert_allclose(
            var_first,
            var_second,
            rtol=1e-6,
            err_msg="Input values mismatch"
        )

        # execute custom op
        custom_out = self.custom_op.kp_fused_sparse_dynamic_stitch(self.x, self.variables)

        # tf native implementation
        tf_out = self._tf_reference_impl(self.x, self.variables)

        custom_out_val = self.sess.run([custom_out])
        tf_out_val = self.sess.run([tf_out])
        print("custom_shape: ")
        print(custom_out_val[0].shape)
        print("tf_out shape: ")
        print(tf_out_val[0].shape)
        # Numerical comparison
        np.testing.assert_allclose(
            custom_out_val[0],
            tf_out_val[0],
            rtol=1e-6,
            err_msg="Output values mismatch"
        )

    def _tf_reference_impl(self, x, variables):
        x_1 = tf.reshape(x, shape=[-1])
        group_ids = tf.math.floormod(x_1, 12)
        group_ids = tf.cast(group_ids, dtype=np.int32)
        chunk_indices = tf.math.floordiv(x_1, 12)

        original_indices = tf.range(0,tf.size(x_1),1)

        a = tf.dynamic_partition(original_indices, group_ids, num_partitions=12)
        b = tf.dynamic_partition(chunk_indices, group_ids, num_partitions=12)

        c = [tf.gather(variables[i], b[i]) for i in range(12)]

        d = tf.dynamic_stitch(a, c)

        return d

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=1)