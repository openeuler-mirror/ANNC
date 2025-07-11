import tensorflow as tf
import numpy as np
import unittest

class TestFusedEmbeddingActionIdGather(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize test data and custom op"""
        # Load custom op
        cls.custom_op = tf.load_op_library('../kernels/fused_embedding_action_id_gather.so')
        
        # Base test data
        np.random.seed(140)
        indices1_shape = (8, 10)
        indices2_shape = (5, 6)
        params_shape = (80, 300)
        cls.input0 = np.random.randint(0, params_shape[0], size=indices1_shape, dtype=np.int32)
        cls.input1 = np.random.random(params_shape).astype(np.float32)
        cls.input2 = np.random.randint(0, indices1_shape[0], size=indices2_shape, dtype=np.int32)
        cls.input3 = params_shape[0]
         # Create tf session
        cls.sess = tf.compat.v1.Session()

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()

    def test_kp_fused_embedding_action_id_gather(self):
        # execute custom op
        custom_out = self.custom_op.kp_fused_embedding_action_id_gather(
            input0=tf.constant(self.input0, dtype=tf.int32),
            input1=tf.constant(self.input1, dtype=tf.float32),
            input2=tf.constant(self.input2, dtype=tf.int32),
            input3=tf.constant(self.input3, dtype=tf.int32),
        )

        # tf native implementation
        tf_out = self._tf_reference_impl(
            input0=tf.constant(self.input0, dtype=tf.int32),
            input1=tf.constant(self.input1, dtype=tf.float32),
            input2=tf.constant(self.input2, dtype=tf.int32),
            input3=tf.constant(self.input3, dtype=tf.int32),
        )

        custom_out_val = self.sess.run([custom_out])
        tf_out_val = self.sess.run([tf_out])
        
        # Numerical comparison
        np.testing.assert_array_equal(
            custom_out_val,
            tf_out_val,
            err_msg="result mismatch"
        )
    
    def _tf_reference_impl(self, input0, input1, input2, input3):
        gather1 = tf.gather(input1, input0, axis=0)
        gather2 = tf.gather(gather1, input2, axis=0)
        pack1 = tf.stack([input3, 1680], axis=0)
        pack2 = tf.stack([input3, -1], axis=0)
        reshape = tf.reshape(gather2, pack2)
        fill = tf.fill(pack1, tf.constant(0, dtype=tf.float32))
        output = tf.concat([reshape, fill], axis=-1)
        return output

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=2)