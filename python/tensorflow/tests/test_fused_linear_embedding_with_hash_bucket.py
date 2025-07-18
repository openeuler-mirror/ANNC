import tensorflow as tf
import numpy as np
import unittest

class TestFusedLinearEmbeddingWithHashBucket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize test data and custom op"""
        # Load custom op
        cls.custom_op = tf.load_op_library('kpfusedlinearembeddingwithhashbucket.so')

        #base test data
        cls.base_placeholder = "test this pattern"
        cls.base_variable = np.linspace(0, 1, num=2, endpoint=False, dtype=np.float32).reshape(2, 1)
         # Create tf session
         cls.sess = tf.compat.v1.session()

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()

    def test_custom(self):
        #execute custom_op
        custom_out = self.custom_op.kpfusedlinearembeddingwithhashbucket(
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
        shape_0_out = tf.shape(expanddims_out)
        notequal_out = tf.not_equal(expanddims_out, y = "")
        cast_0_out = tf.cast(shape_0_out)
        where_0_out = tf.where(not_equal_out)
        stridedslice_0_out = tf.strided_slice(cast_0_out, begin = [0], end = [1], strides = [1])
        pack_0_out = tf.pack(stridedslice_0_out, -1)
        gathernd_out = tf.gather_nd(expanddims_out, where_0_out)
        cast_1_out = tf.cast(pack_0_out)
        stringtohashbucketfast_out = tf.strings.to_hash_bucket_fast(gathernd_out)
        sparsereshape_0_out_outputindices, sparsereshape_0_out_outputshape = tf.sparse_reshape(where_0_out, where_0_out, cast_1_out)
        identity_0_out = tf.identity(stringtohashbucketfast_out)
        slice_0_out = tf.slice(sparsereshape_0_out_outputshape, begin = [0], size = [1])
        identity_1_out = tf.identity(identity_0_out)
        gatherv2_0_out = tf.gather_v2(sparsereshape_0_out_outputshape, indices = 1, axis = 0)
        prod_out = tf.prod(slice_0_out, reduction_indices = [0])
        greaterequal_out = tf.greater_equal(identity_1_out, y = 0)
        pack_1_out = tf.pack(prod_out, gatherv2_0_out)
        where_1_out = tf.where(greaterequal_out)
        sparsereshape_1_out_outputindices, sparsereshape_1_out_outputshape = tf.sparse_reshape(sparsereshape_0_out_outputindices, sparsereshape_0_out_outputshape, pack_1_out)
        reshape_0_out = tf.reshape(where_1_out, shape = [-1])
        gatherv2_1_out = tf.gather_v2(sparsereshape_1_out_outputindices, reshape_0_out, axis = 0)
        gatherv2_2_out = tf.gather_v2(identity_1_out, reshape_0_out, axis = 0)
        identity_2_out = tf.identity(sparsereshape_1_out_outputshape)
        sparsefillemptyrows_out_ouputindice,  sparsefillemptyrows_out_outputvalue,  sparsefillemptyrows_out_emptyrowindicator, sparsefillemptyrows_out_reverseindexmap  = tf.sparse_fill_empty_rows(gatherv2_1_out, gatherv2_2_out, identity_2_out, default_value = 0)
        unique_out_y, unique_out_idx = tf.unique(sparsefillemptyrows_out_outputvalue)
        resourcegather_out = tf.resouce_gather(variable, unique_out_y)
        stridedslice_1_out = tf.strided_slice(sparsefillemptyrows_out_ouputindice, begin = [0,0], end = [0,1], strides = [1,1])
        identity_3_out = tf.identity(resourcegather_out)
        cast_2_out = tf.cast(stridedslice_1_out)
        identity_4_out = tf.identity(identity_3_out)
        sparsesegmentsum_out = tf.sparse_segment_sum(identity_4_out, unique_out_idx, cast_2_out)
        shape_1_out = tf.shape(sparsesegmentsum_out)
        stridedslice_2_out = tf.strided_slice(shape_1_out, begin = [1], end = [2], strides = [1])
        reshape_1_out = tf.reshape(sparsefillemptyrows_out_emptyrowindicator, shape = [-1, 1])
        pack_2_out = tf.pack(1, stridedslice_2_out)
        tile_out = tf.tile(reshape_1_out, pack_2_out)
        zeroslike_out = tf.zeros_like(sparsesegmentsum_out)
        select_out = tf.select(tile_out, zeroslike_out, sparsesegmentsum_out)
        shape_2_out = tf.shape(select_out)
        cast_3_out = tf.cast(sparsereshape_0_out_outputshape)
        slice_1_out = tf.slice(shape_2_out, begin = [1], size = [-1])
        slice_2_out = tf.slice(cast_3_out, begin = [0], size = [1])
        concatv2_out = tf.concat_v2(slice_2_out, slice_1_out, axis = 0)
        reshape_2_out = tf.reshape(select_out, concatv2_out)
        return reshape_2_out

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=2)