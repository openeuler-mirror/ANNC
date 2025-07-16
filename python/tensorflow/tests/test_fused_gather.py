import tensorflow as tf
import numpy as np
import unittest

class TestFusedGather(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize test data and custom op"""
        # Load custom op
        cls.custom_op = tf.load_op_library('kpfusedgather.so')
        
        # Base test data
        cls.base_data = np.linspace(0, 11, num=240, endpoint=False, dtype=np.float32).reshape(20, 12)
        cls.base_slice_input = np.array([[0, 0], [0, 1], [1, 2]], dtype=np.int64)
        cls.base_begin = [0, 1]
        cls.base_end = [0, 2]
        cls.base_strides = [1, 1]
         # Create tf session
        cls.sess = tf.compat.v1.Session()

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()

    def test_custom(self):
        # execute custom op
        custom_out1, custom_out2, custom_out3= self.custom_op.KPFusedGather(
            data=self.base_data,
            slice_input=self.base_slice_input,
            begin=self.base_begin,
        )

        # tf native implementation
        tf_out1, tf_out2, tf_out3 = self._tf_reference_impl(
            self.base_data, 
            self.base_slice_input,
            self.base_begin,
        )

        custom_out_val1, custom_out_val2, custom_out_val3 = self.sess.run([custom_out1, custom_out2, custom_out3])
        tf_out_val1, tf_out_val2, tf_out_val3 = self.sess.run([tf_out1, tf_out2, tf_out3])
        
        print("custom_out_val1: ", custom_out_val1)
        print("custom_out_val2: ", custom_out_val2)
        print("custom_out_val3: ", custom_out_val3)
        print("tf_out_val1: ", tf_out_val1)
        print("tf_out_val2: ", tf_out_val2)
        print("tf_out_val3: ", tf_out_val3)

        np.testing.assert_array_equal(
            custom_out_val1,
            tf_out_val1,
            err_msg="Segment count mismatch"
        )

        np.testing.assert_array_equal(
            custom_out_val2,
            tf_out_val2,
            err_msg="Segment count mismatch"
        )

        np.testing.assert_allclose(
            custom_out_val3,
            tf_out_val3,
            rtol=1e-6,
            err_msg="Output values mismatch"
        )

    def _tf_reference_impl(self, data, slice_input, begin):
        slice_out = tf.strided_slice(
            slice_input,
            begin = begin,
            end = [tf.shape(slice_input)[0], begin[1] + 2],
            strides = [1, 1],
            begin_mask = 1,
            end_mask = 1,
            shrink_axis_mask = 2
        )
        
        slice_out, slice_out_indices = tf.unique(slice_out)
        output_shape = tf.shape(slice_out)
        slice_out = tf.reshape(slice_out, [-1])
        slice_out, _ = tf.unique(slice_out)

        gather1_result = tf.gather(data, slice_out)
        gather1_result = tf.reshape(gather1_result, [-1, 12])

        gather2_result = tf.gather(gather1_result, slice_out)
        return output_shape, slice_out_indices, gather2_result

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=2)