import tensorflow as tf
import numpy as np
import unittest

class TestFusedEmbeddingPadding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize test data and custom op"""
        # Load custom op
        cls.custom_op = tf.load_op_library('../kernels/fused_embedding_padding.so')
        
        # Base test data
        np.random.seed(140)
        cls.input0 = np.random.randint(0, 100, size=(2 * 3, 10), dtype=np.int64)
        cls.input1 = np.random.rand(2 * 2, 10).astype(np.float)
        cls.input2 = cls.input1.shape
        cls.input3 = np.array([-1, 20]).astype(np.int32)
         # Create tf session
        cls.sess = tf.compat.v1.Session()

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()

    def test_kp_fused_embedding_padding_fast(self):
        # execute custom op
        _, custom_out = self.custom_op.kp_fused_embedding_padding_fast(
            input0=self.input0.shape,
            input1=self.input1,
            input2=self.input2[0],
            input3=self.input3,
        )

        # tf native implementation
        tf_out = self._fused_embedding_padding_fast_reference_impl(
            tf.constant(self.input0.shape, dtype=tf.int64),
            tf.constant(self.input1, dtype=tf.float32),
            tf.constant(self.input2[0], dtype=tf.int32),
            tf.constant(self.input3, dtype=tf.int32),
        )

        custom_out_val = self.sess.run([custom_out])
        tf_out_val = self.sess.run([tf_out])
        
        # Numerical comparison
        np.testing.assert_array_equal(
            custom_out_val,
            tf_out_val,
            err_msg="result mismatch"
        )
    
    def test_kp_fused_embedding_padding(self):
        # execute custom op
        _, custom_out = self.custom_op.kp_fused_embedding_padding(
            input0=self.input0.shape,
            input1=self.input1,
            input2=self.input2[0],
            input3=self.input3,
        )

        # tf native implementation
        tf_out = self._fused_embedding_padding_reference_impl(
            tf.constant(self.input0.shape, dtype=tf.int64),
            tf.constant(self.input1, dtype=tf.float32),
            tf.constant(self.input2[0], dtype=tf.int32),
            tf.constant(self.input3, dtype=tf.int32),
        )

        custom_out_val = self.sess.run([custom_out])
        tf_out_val = self.sess.run([tf_out])
        
        # Numerical comparison
        np.testing.assert_array_equal(
            custom_out_val,
            tf_out_val,
            err_msg="result mismatch"
        )

    def _fused_embedding_padding_fast_reference_impl(self, input0, input1, input2, input3):
        cast = tf.cast(input0, tf.int32)
        begin = tf.constant([0], dtype=tf.int32)
        end = tf.constant([1], dtype=tf.int32)
        strides = tf.constant([1], dtype=tf.int32)
        hash_rows = tf.strided_slice(cast, begin=begin, end=end, strides=strides, shrink_axis_mask=1)
        sub_out = hash_rows - input2
        const = tf.constant(10, dtype=tf.int32)
        pack = tf.stack([sub_out, const], axis=0)
        fill = tf.fill(pack, tf.constant(0, dtype=tf.float32))
        concat = tf.concat([input1, fill], 0)
        reshape = tf.reshape(concat, input3)
        shape_tensor = tf.shape(reshape)
        output = tf.strided_slice(shape_tensor, begin=begin, end=end, strides=strides, shrink_axis_mask=1)
        return output

    def _fused_embedding_padding_reference_impl(self, input0, input1, input2, input3):
        cast = tf.cast(input0, tf.int32)
        begin = tf.constant([0], dtype=tf.int32)
        end = tf.constant([1], dtype=tf.int32)
        strides = tf.constant([1], dtype=tf.int32)
        hash_rows = tf.strided_slice(cast, begin=begin, end=end, strides=strides, shrink_axis_mask=1)
        sub_out = hash_rows - input2
        const = tf.constant(10, dtype=tf.int32)
        pack = tf.stack([sub_out, const], axis=0)
        fill = tf.fill(pack, tf.constant(0, dtype=tf.float32))
        concat = tf.concat([input1, fill], 0)
        output = tf.reshape(concat, input3)
        return output

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=2)