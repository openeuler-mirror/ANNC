import tensorflow as tf
import numpy as np
import unittest

class TestSparseSegmentMeanSlice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize test data and custom op"""
        # Load custom op
        cls.custom_op = tf.load_op_library('kppattern6.so')
        
        # Base test data
        cls.base_shape_input = np.array([4, 2, 3, 1], dtype=np.int64)
        cls.base_slice_input = np.array([[0, 0], [0, 1], [1, 2], [3, 4]], dtype=np.int64)
        cls.base_begin = [0, 1]
        cls.base_end = [0, 2]
        cls.base_strides = [1, 1]
        cls.base_newshape = [2, 4]
         # Create tf session
        cls.sess = tf.compat.v1.Session()

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()

    def test_custom(self):
        # execute custom op
        custom_out1, custom_out2, = self.custom_op.KPFusedSparseReshape(
            shape_input=self.base_shape_input,
            slice_input=self.base_slice_input,
            begin=self.base_begin,
            new_shape=self.base_newshape
        )

        # tf native implementation
        tf_out1, tf_out2, tf_out3 = self._tf_reference_impl(
            self.base_shape_input, 
            self.base_slice_input,
            self.base_begin,
            self.base_newshape
        )

        custom_out_val1, custom_out_val2 = self.sess.run([custom_out1, custom_out2])
        tf_out_val1, tf_out_val2, tf_out_val3 = self.sess.run([tf_out1, tf_out2, tf_out3])
        
        print("custom_out_val1: ", custom_out_val1)
        print("custom_out_val2: ", custom_out_val2)
        print("tf_out_val1: ", tf_out_val1)
        print("tf_out_val2: ", tf_out_val2)

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

    def _tf_reference_impl(self, shape_input, slice_input, begin, new_shape):
        slice67_out = tf.strided_slice(
            slice_input,
            begin=begin,
            end=[0, 2],
            strides=[1, 1],
            begin_mask=1,
            end_mask=1,
            shrink_axis_mask=2
        )

        slice67_out = tf.reshape(slice67_out, [-1, 1])
        shape_out = tf.shape(shape_input)
        slice57_out = tf.strided_slice(
            shape_out, 
            begin=[0],
            end=[1],
            strides=[1],
            shrink_axis_mask=1
        )
        
        const2 = tf.constant(2)
        input_shape = tf.stack([slice57_out, const2])
        input_shape = tf.cast(input_shape, tf.int64)

        range_out = tf.range(0, slice57_out, 1)
        range_out = tf.reshape(range_out, [-1, 1])
        range_out_64 = tf.cast(range_out, dtype=tf.int64)
        concat_out = tf.concat([range_out_64, slice67_out], axis=-1)
        
        sparse_tensor = tf.SparseTensor(
            indices=concat_out,
            values=[1,2,3,4],
            dense_shape=input_shape
        )
        sparse_tensor_out = tf.sparse.reshape(sparse_tensor, new_shape)
        return sparse_tensor_out.indices, sparse_tensor_out.dense_shape, concat_out

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=2)