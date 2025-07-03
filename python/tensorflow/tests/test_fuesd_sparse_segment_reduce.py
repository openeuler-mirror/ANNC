import tensorflow as tf
import numpy as np
import unittest

class TestSparseSegmentMeanSlice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize test data and custom op"""
        # Load custom op
        cls.custom_op = tf.load_op_library('kp_fused_sparse_segment_reduce.so')
        
        # Base test data
        cls.base_data = np.array([[1.0, 2.0, 3.0], [3.0, 4.0,5.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]], dtype=np.float32) # shape {4， 3}
        cls.base_indices = np.array([0, 1, 2], dtype=np.int64) # shape {3}
        cls.base_slice_input = np.array([[0, 0], [0, 2], [1, 2]], dtype=np.int64) # shape {3, 2}
        cls.base_begin = [0, 1]
        cls.base_end = [0, 2]
        cls.base_strides = [1, 2]
         # Create tf session
        cls.sess = tf.compat.v1.Session()

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()

    def test_mean(self):
        # execute custom op
        custom_out, custom_slice_out = self.custom_op.kp_fused_sparse_segment_reduce(
            data=self.base_data,
            indices=self.base_indices,
            slice_input=self.base_slice_input,
            begin=self.base_begin,
            end = self.base_end,
            strides = self.base_strides
        )

        # tf native implementation
        tf_out, tf_slice_out = self._tf_reference_impl(
            self.base_data, 
            self.base_indices,
            self.base_slice_input,
            self.base_begin,
            self.base_end,
            self.base_strides,
            True
        )

        custom_out_val, custom_slice_out_val = self.sess.run([custom_out, custom_slice_out])
        tf_out_val, tf_slice_out_val = self.sess.run([tf_out, tf_slice_out])
        
        # Numerical comparison
        np.testing.assert_allclose(
            custom_out_val,
            tf_out_val,
            rtol=1e-6,
            err_msg="Output values mismatch"
        )
        np.testing.assert_array_equal(
            custom_slice_out_val,
            tf_slice_out_val,
            err_msg="Segment count mismatch"
        )
    
    def test_sum(self):
        custom_out, custom_slice_out = self.custom_op.kp_fused_sparse_segment_reduce(
            data=self.base_data,
            indices=self.base_indices,
            slice_input=self.base_slice_input,
            begin=self.base_begin,
            end = self.base_end,
            strides = self.base_strides,
            combiner=0
        )

        tf_out, tf_slice_out = self._tf_reference_impl(
            self.base_data, 
            self.base_indices,
            self.base_slice_input,
            self.base_begin,
            self.base_end,
            self.base_strides,
            False
        )

        custom_out_val, custom_slice_out_val = self.sess.run([custom_out, custom_slice_out])
        tf_out_val, tf_slice_out_val = self.sess.run([tf_out, tf_slice_out])
        
        np.testing.assert_allclose(
            custom_out_val,
            tf_out_val,
            rtol=1e-6,
            err_msg="Output values mismatch"
        )
        np.testing.assert_array_equal(
            custom_slice_out_val,
            tf_slice_out_val,
            err_msg="Segment count mismatch"
        )

    def _tf_reference_impl(self, data, indices, slice_input, begin, end, strides, is_mean):
        slice_out = tf.strided_slice(
            slice_input,
            begin= begin,
            end= end,
            strides= strides,
            begin_mask=1,
            end_mask=1,
            shrink_axis_mask=2
        )
        
        segment_ids = tf.cast(slice_out, dtype=tf.int32)
        if is_mean:
            output = tf.sparse.segment_mean(
                data = data,
                indices = indices,
                segment_ids= segment_ids
            )
        else:
            output = tf.sparse.segment_sum(
                data = data,
                indices = indices,
                segment_ids= segment_ids
            )
        
        output_shape = tf.shape(output)
        slice_out = tf.strided_slice(output_shape, begin=[0], end=[1], strides=[1])
        
        return output, slice_out

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=2)