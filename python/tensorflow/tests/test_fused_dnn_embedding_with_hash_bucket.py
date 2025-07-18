import tensorflow as tf
import numpy as np
import unittest

class TestFusedDnnEmbeddingWithHashBucket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize test data and custom op"""
        # Load custom op
        cls.custom_op = tf.load_op_library('kpfuseddnnembeddingwithhashbucket.so')

        #base test data
        cls.base_placeholder = "test this pattern"
        cls.base_variable = np.linspace(0, 15, num=160000, endpoint=False, dtype=np.float32).reshape(10000, 16)
         # Create tf session
         cls.sess = tf.compat.v1.session()

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()

    def test_custom(self):
        #execute custom_op
        custom_out = self.custom_op.kpfuseddnnembeddingwithhashbucket(
            placeholder = self.base_placeholder,
            variable = self.base_variable,
        )

        # tg native implementation
        tf_out = self._tf_reference_impl(
            self.base_placeholder,
            self.base_variable,
        )

        custom_out_val = self.sess.run([custom_out])
        tf_out_val = self.sess.run([tf_out])

        print("custom_out_val", custom_out_val)
        print("tf_out_val", tf_out_val)

        np.testing.assert_array_equal(
            custom_out_val.
            tf_out_val,
            err_msg = "segment count mismatch"
        )

    def _tf_reference_impl(self, placeholder, variable):
        expanddims_out = tf.expand_dims(placeholder, dims = -1)
        notequal_out = tf.not_equal(expanddims_out, y = "")
        where_0_out = tf.where(notequal_out)
        gathernd_out = tf.gather_nd(expanddims_out, where_0_out)
        stringtohashbucketfast_out = tf.strings.to_hash_bucket_fast(gathernd_out)
        shape_0_out = tf.shape(expanddims_out)
        slice_0_out = tf.slice(shape_0_out, begin = [0], size = [1])
        identity_0_out = tf.identity(stringtohashbucketfast_out)
        gatherv2_0_out = tf.gather_v2(shape_0_out, indices = 1, axis = 0)
        prod_out = tf.prod(slice_0_out, reduction_indices = [0])
        greaterequal_out = tf.greater_equal(identity_0_out, y = 0)
        pack_0_out = tf.pack(prod_out, gatherv2_0_out)
        where_1_out = tf.where(greaterequal_out)
        sparsereshape_out = tf.sparse_reshape(wherepack_0_out, where_0_out, pack_0_out)
        reshape_0_out = tf.reshape(where_1_out, shape = [-1])
        gatherv2_1_out = tf.gather_v2(sparsereshape_out, reshape_0_out, axis = 0,)
        gatherv2_2_out = tf.gather_v2(identity_0_out, reshape_0_out, axis = 0)
        identity_1_out = tf.identity(sparsereshape_out)
        sparsefillemptyrows_out_ouputindice,  sparsefillemptyrows_out_outputvalue,  sparsefillemptyrows_out_emptyrowindicator, sparsefillemptyrows_out_reverseindexmap  = tf.sparse_fill_empty_rows(gatherv2_1_out, gatherv2_2_out, identity_1_out, default_value = 0)
        unique_out_y, unique_out_idx = tf.unique(sparsefillemptyrows_out_outputvalue)
        stridedslice_0_out = tf.strided_slice(sparsefillemptyrows_out_ouputindice, begin = [0, 0], end = [0, 1], strides = [1, 1])
        gatherv2_3_out = tf.gather_v2(variable ,unique_out_y, axis = 0)
        cast_0_out = tf.cast(stridedslice_0_out)
        identity_2_out = tf.identity(gatherv2_3_out)
        sparsesegmentmean_out = tf.sparse_segment_mean(identity_2_out, unique_out_idx, cast_0_out)
        shape_1_out = tf.shape(sparsesegmentmean_out)
        stridedslice_1_out = tf.strided_slice(shape_1_out, begin = [1], end = [2], strides = [1])
        reshape_1_out = tf.reshape(sparsefillemptyrows_out_emptyrowindicator, shape = [-1, 1])
        pack_1_out = tf.pack(1, stridedslice_1_out)
        tile_out = tf.tile(reshape_1_out, pack_1_out)
        zeroslike_out = tf.zeros_like(sparsesegmentmean_out)
        selec_out = tf.select(tile_out, zeroslike_out, sparsesegmentmean_out)
        shape_2_out = tf.shape(selec_out)
        cast_1_out = tf.cast(shape_0_out)
        slice_1_out = tf.slice(shape_2_out, begin = [1], size = [-1])
        slice_2_out = tf.slice(cast_1_out, begin = [0], size = [1])
        concatv2_out = tf.concat_v2(slice_2_out, slice_1_out, axis = 0)
        reshape_2_out = tf.reshape(selec_out, concatv2_out)
        shape_3_out = tf.shape(reshape_2_out)
        stridedslice_2_out = tf.strided_slice(shape_3_out, begin = [0], end = [1], strides = [1])
        pack_2_out = tf.pack(stridedslice_2_out, 16)
        reshape_3_out =tf.reshape(reshape_2_out, pack_2_out)
        return reshape_3_out

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=2)