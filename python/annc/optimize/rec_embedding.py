from .graph import Node, CustomNode, CustomAttr
from .rewriter import BaseRewriter, CheckFailed
from .op_type import OpType


class DnnSparseEmbeddingPatternRewriter(BaseRewriter):

    def match_and_rewrite(self, node: Node):
        self.check_node(node, (OpType.ExpandDims, None))
        self.check_operands(node, [(OpType.Placeholder, None),
                                   (OpType.Const, None)])
        expand_dim = node.operands[1][0]
        self.check_users(node,
                         [(OpType.NotEqual, None), (OpType.GatherNd, None),
                          (OpType.Shape, None)], 3)
        not_equal, gather_nd, shape_op = node.users
        self.check_users(not_equal, [(OpType.Where, None)], 1)
        where_op = not_equal.users[0]
        self.check_users(where_op, [(OpType.GatherNd, gather_nd.name),
                                    (OpType.SparseReshape, None)])
        sparse_reshape = where_op.users[1]
        self.check_users(gather_nd, [(OpType.StringToHashBucketFast, None)], 1)
        string_to_hash_bucket_fast = gather_nd.users[0]
        self.check_users(string_to_hash_bucket_fast, [(OpType.Identity, None)],
                         1)
        identity = string_to_hash_bucket_fast.users[0]
        self.check_users(identity, [(OpType.GreaterEqual, None),
                                    (OpType.GatherV2, None)], 2)
        greater_equal, gather_v2 = identity.users
        self.check_users(greater_equal, [(OpType.Where, None)], 1)
        where_op_2 = greater_equal.users[0]
        self.check_users(where_op_2, [(OpType.Reshape, None)], 1)
        reshape = where_op_2.users[0]
        self.check_users(reshape, [(OpType.GatherV2, None),
                                   (OpType.GatherV2, gather_v2.name)], 2)
        gather_v2_1, gather_v2_2 = reshape.users
        self.check_users(shape_op,
                         [(OpType.Slice, None), (OpType.GatherV2, None),
                          (OpType.SparseReshape, sparse_reshape.name),
                          (OpType.Cast, None)], 4)
        slice_op, gather_v2_3 = shape_op.users[:2]
        self.check_users(slice_op, [(OpType.Prod, None)], 1)
        prod_op = slice_op.users[0]
        self.check_users(prod_op, [(OpType.Pack, None)], 1)
        pack_op = prod_op.users[0]
        self.check_users(gather_v2_3, [(OpType.Pack, pack_op.name)], 1)
        self.check_users(pack_op,
                         [(OpType.SparseReshape, sparse_reshape.name)], 1)
        self.check_users(sparse_reshape, [(OpType.GatherV2, gather_v2_1.name),
                                          (OpType.Identity, None)], 2)
        identity_2 = sparse_reshape.users[1]
        self.check_users(identity_2, [(OpType.SparseFillEmptyRows, None)], 1)
        self.check_users(gather_v2, [(OpType.SparseFillEmptyRows, None)], 1)
        self.check_users(gather_v2_1, [(OpType.SparseFillEmptyRows, None)], 1)
        sparse_fill_empty_rows = identity_2.users[0]

        index = node.get_index()
        self.graph.nodes.insert(
            index + 1,
            CustomNode(
                'KPFusedSparseEmbedding',
                sparse_fill_empty_rows.name + '/rec_embed_kp_sparse',
                self.graph, sparse_fill_empty_rows.output_shapes[:3] +
                shape_op.output_shapes, node.operands[:1],
                string_to_hash_bucket_fast.attrs,
                sparse_fill_empty_rows.users + shape_op.users))

        self.replace_all_users_with(shape_op, 0, self.graph.nodes[index + 1],
                                    3)
        self.replace_all_users_with(sparse_fill_empty_rows, 0,
                                    self.graph.nodes[index + 1], 0)
        self.replace_all_users_with(sparse_fill_empty_rows, 1,
                                    self.graph.nodes[index + 1], 1)
        self.replace_all_users_with(sparse_fill_empty_rows, 2,
                                    self.graph.nodes[index + 1], 2)

        fused_ops = [
            expand_dim, node, not_equal, where_op, gather_nd,
            string_to_hash_bucket_fast, identity, greater_equal, where_op_2,
            reshape, gather_v2, gather_v2_1, shape_op, slice_op, prod_op,
            gather_v2_3, pack_op, sparse_reshape, identity_2,
            sparse_fill_empty_rows
        ]
        for fused_op in fused_ops:
            self.graph.delete_node(fused_op)


class DnnEmbeddingWithHashBucketPatternRewriter(BaseRewriter):

    def match_and_rewrite(self, node: Node):
        self.check_node(node, (OpType.ExpandDims, None))
        self.check_operands(node, [(OpType.Placeholder, None),
                                   (OpType.Const, None)])
        expand_dim = node.operands[1][0]
        self.check_users(node,
                         [(OpType.NotEqual, None), (OpType.GatherNd, None),
                          (OpType.Shape, None)], 3)
        not_equal, gather_nd, shape_op = node.users
        self.check_users(not_equal, [(OpType.Where, None)], 1)
        where_op = not_equal.users[0]
        self.check_users(where_op, [(OpType.GatherNd, gather_nd.name),
                                    (OpType.SparseReshape, None)])
        sparse_reshape = where_op.users[1]
        self.check_users(gather_nd, [(OpType.StringToHashBucketFast, None)], 1)
        string_to_hash_bucket_fast = gather_nd.users[0]
        self.check_users(string_to_hash_bucket_fast, [(OpType.Identity, None)],
                         1)
        identity = string_to_hash_bucket_fast.users[0]
        self.check_users(identity, [(OpType.GreaterEqual, None),
                                    (OpType.GatherV2, None)], 2)
        greater_equal, gather_v2 = identity.users
        self.check_users(greater_equal, [(OpType.Where, None)], 1)
        where_op_2 = greater_equal.users[0]
        self.check_users(where_op_2, [(OpType.Reshape, None)], 1)
        reshape = where_op_2.users[0]
        self.check_users(reshape, [(OpType.GatherV2, None),
                                   (OpType.GatherV2, gather_v2.name)], 2)
        gather_v2_1, gather_v2_2 = reshape.users
        self.check_users(shape_op,
                         [(OpType.Slice, None), (OpType.GatherV2, None),
                          (OpType.SparseReshape, sparse_reshape.name),
                          (OpType.Cast, None)], 4)
        slice_op, gather_v2_3 = shape_op.users[:2]
        self.check_users(slice_op, [(OpType.Prod, None)], 1)
        prod_op = slice_op.users[0]
        self.check_users(prod_op, [(OpType.Pack, None)], 1)
        pack_op = prod_op.users[0]
        self.check_users(gather_v2_3, [(OpType.Pack, pack_op.name)], 1)
        self.check_users(pack_op,
                         [(OpType.SparseReshape, sparse_reshape.name)], 1)
        self.check_users(sparse_reshape, [(OpType.GatherV2, gather_v2_1.name),
                                          (OpType.Identity, None)], 2)
        identity_2 = sparse_reshape.users[1]
        self.check_users(identity_2, [(OpType.SparseFillEmptyRows, None)], 1)
        self.check_users(gather_v2, [(OpType.SparseFillEmptyRows, None)], 1)
        self.check_users(gather_v2_1, [(OpType.SparseFillEmptyRows, None)], 1)
        sparse_fill_empty_rows = identity_2.users[0]

        self.check_users(sparse_fill_empty_rows, [(OpType.StridedSlice, None),
                                                  (OpType.Unique, None),
                                                  (OpType.Reshape, None)], 3)
        stride_slice, unique_op, reshape_op = sparse_fill_empty_rows.users
        self.check_users(reshape_op, [(OpType.Tile, None)], 1)
        self.check_users(unique_op, [(OpType.GatherV2, None),
                                     (OpType.SparseSegmentMean, None)], 2)
        gather_v2_4, sparse_segment_mean = unique_op.users
        self.check_users(gather_v2_4, [(OpType.Identity, None)], 1)
        identity_3 = gather_v2_4.users[0]
        self.check_users(identity_3, [(OpType.SparseSegmentMean, None)], 1)
        self.check_users(stride_slice, [(OpType.Cast, None)], 1)
        cast_op = stride_slice.users[0]
        self.check_users(cast_op, [(OpType.SparseSegmentMean, None)], 1)
        self.check_users(sparse_segment_mean, [(OpType.Shape, None),
                                               (OpType.ZerosLike, None),
                                               (OpType.Select, None)], 3)
        shape_op_2, zeros_like, select_op = sparse_segment_mean.users
        self.check_users(zeros_like, [(OpType.Select, None)], 1)
        self.check_users(shape_op_2, [(OpType.StridedSlice, None)], 1)
        stride_slice_2 = shape_op_2.users[0]
        self.check_users(stride_slice_2, [(OpType.Pack, None)], 1)
        pack_op_2 = stride_slice_2.users[0]
        self.check_users(pack_op_2, [(OpType.Tile, None)], 1)
        tile_op = pack_op_2.users[0]
        self.check_users(tile_op, [(OpType.Select, None)], 1)
        self.check_users(select_op, [(OpType.Shape, None),
                                     (OpType.Reshape, None)], 2)
        shape_op_3, reshape_op_2 = select_op.users
        self.check_users(shape_op_3, [(OpType.Slice, None)], 1)
        slice_op_2 = shape_op_3.users[0]
        self.check_users(slice_op_2, [(OpType.ConcatV2, None)], 1)
        cast_op_2 = shape_op.users[3]
        self.check_users(cast_op_2, [(OpType.Slice, None)], 1)
        slice_op_3 = cast_op_2.users[0]
        self.check_users(slice_op_3, [(OpType.ConcatV2, None)], 1)
        concatv2 = slice_op_3.users[0]
        self.check_users(concatv2, [(OpType.Reshape, None)], 1)
        self.check_users(reshape_op_2, [(OpType.Shape, None),
                                        (OpType.Reshape, None)], 2)
        shape_op_4, reshape_op_3 = reshape_op_2.users
        self.check_users(shape_op_4, [(OpType.StridedSlice, None)], 1)
        stride_slice_3 = shape_op_4.users[0]
        self.check_users(stride_slice_3, [(OpType.Pack, None)], 1)
        pack_op_3 = stride_slice_3.users[0]
        self.check_users(pack_op_3, [(OpType.Reshape, None)], 1)

        index = node.get_index()
        self.graph.nodes.insert(
            index + 1,
            CustomNode(
                'KPFusedDnnEmbeddingWithHashBucket',
                sparse_fill_empty_rows.name + '/rec_embed_kp_dnn_bucket',
                self.graph, reshape_op_3.output_shapes,
                node.operands[:1] + gather_v2_4.operands[:1],
                string_to_hash_bucket_fast.attrs, reshape_op_3.users))
        self.replace_all_users_with(reshape_op_3, 0,
                                    self.graph.nodes[index + 1], 0)

        fused_ops = [
            expand_dim, node, not_equal, where_op, gather_nd,
            string_to_hash_bucket_fast, identity, greater_equal, where_op_2,
            reshape, gather_v2, gather_v2_1, shape_op, slice_op, prod_op,
            gather_v2_3, pack_op, sparse_reshape, identity_2,
            sparse_fill_empty_rows, unique_op, gather_v2_4, identity_3,
            stride_slice, cast_op, sparse_segment_mean, shape_op_2,
            stride_slice_2, pack_op_2, reshape_op, tile_op, zeros_like,
            select_op, shape_op_3, slice_op_2, concatv2, cast_op_2, slice_op_3,
            reshape_op_2, shape_op_4, stride_slice_3, pack_op_3, reshape_op_3
        ]
        for fused_op in fused_ops:
            self.graph.delete_node(fused_op)


class EmbeddingPatternRewriter(BaseRewriter):

    def match_seg_sum(self, node: Node):
        self.check_node(node, (OpType.SparseFillEmptyRows, None))
        self.check_operands(node, [(OpType.GatherV2, None),
                                   (OpType.GatherV2, None),
                                   (OpType.Identity, None)])
        gather_v2_0: Node = node.operands[0][0]
        gather_v2_1: Node = node.operands[1][0]
        identity_0: Node = node.operands[2][0]
        self.check_operands(gather_v2_0, [(OpType.SparseReshape, None),
                                          (OpType.Reshape, None)])
        sparse_reshape_0: Node = gather_v2_0.operands[0][0]
        reshape_0: Node = gather_v2_0.operands[1][0]
        self.check_operands(gather_v2_1, [(OpType.Identity, None),
                                          (OpType.Reshape, reshape_0.name)])
        identity_1: Node = gather_v2_1.operands[0][0]
        self.check_operands(identity_0,
                            [(OpType.SparseReshape, sparse_reshape_0.name)])
        self.check_operands(reshape_0, [(OpType.Where, None)])
        where_0: Node = reshape_0.operands[0][0]
        self.check_operands(where_0, [(OpType.GreaterEqual, None)])
        greater_equal_0 = where_0.operands[0][0]
        self.check_operands(greater_equal_0,
                            [(OpType.Identity, identity_1.name)])
        self.check_operands(identity_1, [(OpType.Identity, None)])
        identity_2: Node = identity_1.operands[0][0]
        self.check_operands(identity_2,
                            [(OpType.StringToHashBucketFast, None)])

        self.check_operands(sparse_reshape_0, [(OpType.SparseReshape, None),
                                               (OpType.SparseReshape, None),
                                               (OpType.Pack, None)])
        sparse_reshape_1: Node = sparse_reshape_0.operands[0][0]
        pack_0: Node = sparse_reshape_0.operands[2][0]
        self.check_operands(pack_0, [(OpType.Prod, None),
                                     (OpType.GatherV2, None)])
        prod_0: Node = pack_0.operands[0][0]
        gather_v2_2: Node = pack_0.operands[1][0]
        self.check_operands(prod_0, [(OpType.Slice, None)])
        slice_0 = prod_0.operands[0][0]
        self.check_operands(slice_0,
                            [(OpType.SparseReshape, sparse_reshape_1.name)])
        self.check_operands(gather_v2_2,
                            [(OpType.SparseReshape, sparse_reshape_1.name)])
        self.check_operands(sparse_reshape_1, [(OpType.Where, None),
                                               (OpType.Shape, None),
                                               (OpType.Cast, None)])
        where_1: Node = sparse_reshape_1.operands[0][0]
        shape_0: Node = sparse_reshape_1.operands[1][0]
        cast_0: Node = sparse_reshape_1.operands[2][0]
        self.check_operands(cast_0, [(OpType.Pack, None)])
        pack_1: Node = cast_0.operands[0][0]
        self.check_operands(pack_1, [(OpType.StridedSlice, None)])
        stride_slice_0: Node = pack_1.operands[0][0]
        cast_1 = stride_slice_0.operands[0][0]
        self.check_operands(cast_1, [(OpType.Shape, shape_0.name)])

        self.check_users(node, [(OpType.StridedSlice, None),
                                (OpType.Unique, None), (OpType.Reshape, None)],
                         3)
        stride_slice_1: Node = node.users[0]
        unique_0: Node = node.users[1]
        reshape_1: Node = node.users[2]
        self.check_users(unique_0, [(OpType.ResourceGather, None),
                                    (OpType.SparseSegmentSum, None)], 2)
        resource_gather_0: Node = unique_0.users[0]
        sparse_segment_sum_0: Node = unique_0.users[1]
        self.check_users(resource_gather_0, [(OpType.Identity, None)])
        identity_3: Node = resource_gather_0.users[0]
        self.check_users(identity_3, [(OpType.Identity, None)])
        identity_4: Node = identity_3.users[0]
        self.check_users(
            identity_4, [(OpType.SparseSegmentSum, sparse_segment_sum_0.name)])
        self.check_users(stride_slice_1, [(OpType.Cast, None)])
        cast_2: Node = stride_slice_1.users[0]
        self.check_users(
            cast_2, [(OpType.SparseSegmentSum, sparse_segment_sum_0.name)])
        self.check_users(sparse_segment_sum_0, [(OpType.Shape, None),
                                                (OpType.ZerosLike, None),
                                                (OpType.Select, None)])
        shape_1: Node = sparse_segment_sum_0.users[0]
        zeros_like_0: Node = sparse_segment_sum_0.users[1]
        select_0: Node = sparse_segment_sum_0.users[2]
        self.check_users(shape_1, [(OpType.StridedSlice, None)])
        stride_slice_2: Node = shape_1.users[0]
        self.check_users(stride_slice_2, [(OpType.Pack, None)])
        pack_2: Node = stride_slice_2.users[0]
        self.check_users(pack_2, [(OpType.Tile, None)])
        tile_0: Node = pack_2.users[0]
        self.check_users(reshape_1, [(OpType.Tile, tile_0.name)])
        self.check_users(tile_0, [(OpType.Select, select_0.name)])
        self.check_users(zeros_like_0, [(OpType.Select, select_0.name)])

        self.check_users(select_0, [(OpType.Shape, None),
                                    (OpType.Reshape, None)])
        shape_2: Node = select_0.users[0]
        reshape_2: Node = select_0.users[1]
        self.check_users(shape_2, [(OpType.Slice, None)])
        slice_1: Node = shape_2.users[0]
        self.check_users(slice_1, [(OpType.ConcatV2, None)])
        concat_v2_0: Node = slice_1.users[0]
        self.check_users(sparse_reshape_1, [(OpType.Slice, None),
                                            (OpType.GatherV2, None),
                                            (OpType.SparseReshape, None),
                                            (OpType.SparseReshape, None),
                                            (OpType.Cast, None)])
        cast_3: Node = sparse_reshape_1.users[4]
        self.check_users(cast_3, [(OpType.Slice, None)])
        slice_2: Node = cast_3.users[0]
        self.check_users(slice_2, [(OpType.ConcatV2, concat_v2_0.name)])
        self.check_users(concat_v2_0, [(OpType.Reshape, reshape_2.name)])

        # combiner: 0 for sum, 1 for mean
        custom_node = CustomNode(
            'KPFusedEmbedding', node.name + '/rec_embed_kp', self.graph,
            reshape_2.output_shapes, [
                resource_gather_0.operands[0], identity_2.operands[0],
                sparse_reshape_1.operands[1], sparse_reshape_1.operands[0]
            ], [
                CustomAttr('combiner', 'i: 0'),
                CustomAttr('T_weight', 'type: DT_RESOURCE')
            ], reshape_2.users)
        fused_ops = [
            cast_1, stride_slice_0, pack_1, cast_0, sparse_reshape_1, slice_0,
            prod_0, gather_v2_2, pack_0, sparse_reshape_0, identity_2,
            identity_1, greater_equal_0, where_0, reshape_0, gather_v2_0,
            gather_v2_1, identity_0, node, unique_0, resource_gather_0,
            identity_3, identity_4, stride_slice_1, cast_2,
            sparse_segment_sum_0, shape_1, stride_slice_2, pack_2, reshape_1,
            tile_0, zeros_like_0, select_0, shape_2, slice_1, cast_3, slice_2,
            concat_v2_0, reshape_2
        ]
        replace_ops = [(reshape_2, 0)]
        return custom_node, fused_ops, replace_ops

    def match_seg_mean(self, node: Node):
        self.check_node(node, (OpType.SparseFillEmptyRows, None))
        self.check_operands(node, [(OpType.GatherV2, None),
                                   (OpType.GatherV2, None),
                                   (OpType.Identity, None)])
        gather_v2_0: Node = node.operands[0][0]
        gather_v2_1: Node = node.operands[1][0]
        identity_0: Node = node.operands[2][0]
        self.check_operands(gather_v2_0, [(OpType.SparseReshape, None),
                                          (OpType.Reshape, None)])
        sparse_reshape_0: Node = gather_v2_0.operands[0][0]
        reshape_0: Node = gather_v2_0.operands[1][0]
        self.check_operands(gather_v2_1, [(OpType.Identity, None),
                                          (OpType.Reshape, reshape_0.name)])
        identity_1: Node = gather_v2_1.operands[0][0]
        self.check_operands(identity_0,
                            [(OpType.SparseReshape, sparse_reshape_0.name)])
        self.check_operands(reshape_0, [(OpType.Where, None)])
        where_0: Node = reshape_0.operands[0][0]
        self.check_operands(where_0, [(OpType.GreaterEqual, None)])
        greater_equal_0 = where_0.operands[0][0]
        self.check_operands(greater_equal_0,
                            [(OpType.Identity, identity_1.name)])
        self.check_operands(identity_1,
                            [(OpType.StringToHashBucketFast, None)])
        self.check_operands(sparse_reshape_0, [(OpType.Where, None),
                                               (OpType.Shape, None),
                                               (OpType.Pack, None)])
        shape_0: Node = sparse_reshape_0.operands[1][0]
        pack_0: Node = sparse_reshape_0.operands[2][0]
        self.check_operands(pack_0, [(OpType.Prod, None),
                                     (OpType.GatherV2, None)])
        prod_0: Node = pack_0.operands[0][0]
        gather_v2_2: Node = pack_0.operands[1][0]
        self.check_operands(prod_0, [(OpType.Slice, None)])
        slice_0 = prod_0.operands[0][0]
        self.check_operands(slice_0, [(OpType.Shape, None)])
        self.check_operands(gather_v2_2, [(OpType.Shape, None)])

        self.check_users(node, [(OpType.StridedSlice, None),
                                (OpType.Unique, None), (OpType.Reshape, None)],
                         3)
        stride_slice_1: Node = node.users[0]
        unique_0: Node = node.users[1]
        reshape_1: Node = node.users[2]
        self.check_users(unique_0, [(OpType.GatherV2, None),
                                    (OpType.SparseSegmentMean, None)])
        gather_v2_3: Node = unique_0.users[0]
        sparse_segment_mean_0: Node = unique_0.users[1]
        self.check_users(gather_v2_3, [(OpType.Identity, None)])
        identity_3: Node = gather_v2_3.users[0]
        self.check_users(
            identity_3,
            [(OpType.SparseSegmentMean, sparse_segment_mean_0.name)])
        self.check_users(stride_slice_1, [(OpType.Cast, None)])
        cast_2: Node = stride_slice_1.users[0]
        self.check_users(
            cast_2, [(OpType.SparseSegmentMean, sparse_segment_mean_0.name)])
        self.check_users(sparse_segment_mean_0, [(OpType.Shape, None),
                                                 (OpType.ZerosLike, None),
                                                 (OpType.Select, None)])
        shape_1: Node = sparse_segment_mean_0.users[0]
        zeros_like_0: Node = sparse_segment_mean_0.users[1]
        select_0: Node = sparse_segment_mean_0.users[2]
        self.check_users(shape_1, [(OpType.StridedSlice, None)])
        stride_slice_2: Node = shape_1.users[0]
        self.check_users(stride_slice_2, [(OpType.Pack, None)])
        pack_1: Node = stride_slice_2.users[0]
        self.check_users(pack_1, [(OpType.Tile, None)])
        tile_0: Node = pack_1.users[0]
        self.check_users(reshape_1, [(OpType.Tile, tile_0.name)])
        self.check_users(tile_0, [(OpType.Select, select_0.name)])
        self.check_users(zeros_like_0, [(OpType.Select, select_0.name)])

        self.check_users(select_0, [(OpType.Shape, None),
                                    (OpType.Reshape, None)])
        shape_2: Node = select_0.users[0]
        reshape_2: Node = select_0.users[1]
        self.check_users(shape_2, [(OpType.Slice, None)])
        slice_1: Node = shape_2.users[0]
        self.check_users(slice_1, [(OpType.ConcatV2, None)])
        concat_v2_0: Node = slice_1.users[0]
        self.check_users(shape_0, [(OpType.Slice, None),
                                   (OpType.GatherV2, None),
                                   (OpType.SparseReshape, None),
                                   (OpType.Cast, None)])
        cast_3: Node = shape_0.users[3]
        self.check_users(cast_3, [(OpType.Slice, None)])
        slice_2: Node = cast_3.users[0]
        self.check_users(slice_2, [(OpType.ConcatV2, concat_v2_0.name)])
        self.check_users(concat_v2_0, [(OpType.Reshape, reshape_2.name)])
        self.check_users(reshape_2, [(OpType.Shape, None),
                                     (OpType.Reshape, None)])
        shape_3: Node = reshape_2.users[0]
        reshape_3: Node = reshape_2.users[1]
        self.check_users(shape_3, [(OpType.StridedSlice, None)])
        stride_slice_3: Node = shape_3.users[0]
        self.check_users(stride_slice_3, [(OpType.Pack, None)])
        pack_2: Node = stride_slice_3.users[0]
        self.check_users(pack_2, [(OpType.Reshape, reshape_3.name)])

        # combiner: 0 for sum, 1 for mean
        custom_node = CustomNode(
            'KPFusedEmbedding', node.name + '/rec_embed_kp', self.graph,
            reshape_3.output_shapes, [
                gather_v2_3.operands[0], identity_1.operands[0],
                sparse_reshape_0.operands[1], sparse_reshape_0.operands[0]
            ], [
                CustomAttr('combiner', 'i: 1'),
                CustomAttr('T_weight', 'type: DT_FLOAT')
            ], reshape_3.users)
        fused_ops = [
            slice_0, prod_0, gather_v2_2, pack_0, sparse_reshape_0, identity_1,
            greater_equal_0, where_0, reshape_0, identity_0, gather_v2_1,
            gather_v2_0, node, unique_0, gather_v2_3, identity_3,
            stride_slice_1, cast_2, sparse_segment_mean_0, shape_1,
            stride_slice_2, pack_1, tile_0, zeros_like_0, select_0, shape_2,
            slice_1, cast_3, slice_2, concat_v2_0, reshape_2, shape_3,
            stride_slice_3, pack_2, reshape_3, reshape_1
        ]
        replace_ops = [(reshape_3, 0)]
        return custom_node, fused_ops, replace_ops

    def match_and_rewrite(self, node: Node):
        try:
            res = self.match_seg_sum(node)
        except CheckFailed:
            res = None
        if res is None:
            res = self.match_seg_mean(node)
        custom_node, fused_ops, replace_ops = res

        index = node.get_index()
        self.graph.nodes.insert(index + 1, custom_node)
        for rep_op in replace_ops:
            self.replace_all_users_with(*rep_op, self.graph.nodes[index + 1],
                                        0)
        for fused_op in fused_ops:
            self.graph.delete_node(fused_op)


class EmbeddingWithHashBucketPatternRewriter(BaseRewriter):

    def match_seg_sum(self, node: Node):
        self.check_node(node, (OpType.SparseFillEmptyRows, None))
        self.check_operands(node, [(OpType.GatherV2, None),
                                   (OpType.GatherV2, None),
                                   (OpType.Identity, None)])
        gather_v2_0: Node = node.operands[0][0]
        gather_v2_1: Node = node.operands[1][0]
        identity_0: Node = node.operands[2][0]
        self.check_operands(gather_v2_0, [(OpType.SparseReshape, None),
                                          (OpType.Reshape, None)])
        sparse_reshape_0: Node = gather_v2_0.operands[0][0]
        reshape_0: Node = gather_v2_0.operands[1][0]
        self.check_operands(gather_v2_1, [(OpType.Identity, None),
                                          (OpType.Reshape, reshape_0.name)])
        identity_1: Node = gather_v2_1.operands[0][0]
        self.check_operands(identity_0,
                            [(OpType.SparseReshape, sparse_reshape_0.name)])
        self.check_operands(reshape_0, [(OpType.Where, None)])
        where_0: Node = reshape_0.operands[0][0]
        self.check_operands(where_0, [(OpType.GreaterEqual, None)])
        greater_equal_0 = where_0.operands[0][0]
        self.check_operands(greater_equal_0,
                            [(OpType.Identity, identity_1.name)])
        self.check_operands(identity_1, [(OpType.Identity, None)])
        identity_2: Node = identity_1.operands[0][0]
        self.check_operands(identity_2,
                            [(OpType.StringToHashBucketFast, None)])

        self.check_operands(sparse_reshape_0, [(OpType.SparseReshape, None),
                                               (OpType.SparseReshape, None),
                                               (OpType.Pack, None)])
        sparse_reshape_1: Node = sparse_reshape_0.operands[0][0]
        pack_0: Node = sparse_reshape_0.operands[2][0]
        self.check_operands(pack_0, [(OpType.Prod, None),
                                     (OpType.GatherV2, None)])
        prod_0: Node = pack_0.operands[0][0]
        gather_v2_2: Node = pack_0.operands[1][0]
        self.check_operands(prod_0, [(OpType.Slice, None)])
        slice_0 = prod_0.operands[0][0]
        self.check_operands(slice_0,
                            [(OpType.SparseReshape, sparse_reshape_1.name)])
        self.check_operands(gather_v2_2,
                            [(OpType.SparseReshape, sparse_reshape_1.name)])
        self.check_operands(sparse_reshape_1, [(OpType.Where, None),
                                               (OpType.Shape, None),
                                               (OpType.Cast, None)])
        where_1: Node = sparse_reshape_1.operands[0][0]
        shape_0: Node = sparse_reshape_1.operands[1][0]
        cast_0: Node = sparse_reshape_1.operands[2][0]
        self.check_operands(cast_0, [(OpType.Pack, None)])
        pack_1: Node = cast_0.operands[0][0]
        self.check_operands(pack_1, [(OpType.StridedSlice, None)])
        stride_slice_0: Node = pack_1.operands[0][0]
        cast_1 = stride_slice_0.operands[0][0]
        self.check_operands(cast_1, [(OpType.Shape, shape_0.name)])
        self.check_operands(where_1, [(OpType.NotEqual, None)])
        not_equal: Node = where_1.operands[0][0]
        self.check_operands(not_equal, [(OpType.ExpandDims, None)])
        expand_dims: Node = not_equal.operands[0][0]
        string_to_hash: Node = identity_2.operands[0][0]
        self.check_operands(string_to_hash, [(OpType.GatherNd, None)])
        gather_nd: Node = string_to_hash.operands[0][0]
        self.check_operands(gather_nd, [(OpType.ExpandDims, expand_dims.name),
                                        (OpType.Where, where_1.name)])

        self.check_users(node, [(OpType.StridedSlice, None),
                                (OpType.Unique, None), (OpType.Reshape, None)],
                         3)
        stride_slice_1: Node = node.users[0]
        unique_0: Node = node.users[1]
        reshape_1: Node = node.users[2]
        self.check_users(unique_0, [(OpType.ResourceGather, None),
                                    (OpType.SparseSegmentSum, None)], 2)
        resource_gather_0: Node = unique_0.users[0]
        sparse_segment_sum_0: Node = unique_0.users[1]
        self.check_users(resource_gather_0, [(OpType.Identity, None)])
        identity_3: Node = resource_gather_0.users[0]
        self.check_users(identity_3, [(OpType.Identity, None)])
        identity_4: Node = identity_3.users[0]
        self.check_users(
            identity_4, [(OpType.SparseSegmentSum, sparse_segment_sum_0.name)])
        self.check_users(stride_slice_1, [(OpType.Cast, None)])
        cast_2: Node = stride_slice_1.users[0]
        self.check_users(
            cast_2, [(OpType.SparseSegmentSum, sparse_segment_sum_0.name)])
        self.check_users(sparse_segment_sum_0, [(OpType.Shape, None),
                                                (OpType.ZerosLike, None),
                                                (OpType.Select, None)])
        shape_1: Node = sparse_segment_sum_0.users[0]
        zeros_like_0: Node = sparse_segment_sum_0.users[1]
        select_0: Node = sparse_segment_sum_0.users[2]
        self.check_users(shape_1, [(OpType.StridedSlice, None)])
        stride_slice_2: Node = shape_1.users[0]
        self.check_users(stride_slice_2, [(OpType.Pack, None)])
        pack_2: Node = stride_slice_2.users[0]
        self.check_users(pack_2, [(OpType.Tile, None)])
        tile_0: Node = pack_2.users[0]
        self.check_users(reshape_1, [(OpType.Tile, tile_0.name)])
        self.check_users(tile_0, [(OpType.Select, select_0.name)])
        self.check_users(zeros_like_0, [(OpType.Select, select_0.name)])

        self.check_users(select_0, [(OpType.Shape, None),
                                    (OpType.Reshape, None)])
        shape_2: Node = select_0.users[0]
        reshape_2: Node = select_0.users[1]
        self.check_users(shape_2, [(OpType.Slice, None)])
        slice_1: Node = shape_2.users[0]
        self.check_users(slice_1, [(OpType.ConcatV2, None)])
        concat_v2_0: Node = slice_1.users[0]
        self.check_users(sparse_reshape_1, [(OpType.Slice, None),
                                            (OpType.GatherV2, None),
                                            (OpType.SparseReshape, None),
                                            (OpType.SparseReshape, None),
                                            (OpType.Cast, None)])
        cast_3: Node = sparse_reshape_1.users[4]
        self.check_users(cast_3, [(OpType.Slice, None)])
        slice_2: Node = cast_3.users[0]
        self.check_users(slice_2, [(OpType.ConcatV2, concat_v2_0.name)])
        self.check_users(concat_v2_0, [(OpType.Reshape, reshape_2.name)])

        # combiner: 0 for sum, 1 for mean
        custom_node = CustomNode(
            'KPFusedEmbeddingWithHashBucket', node.name + '/rec_embed_kp_bucket',
            self.graph, reshape_2.output_shapes,
            expand_dims.operands[:1] + resource_gather_0.operands[:1], [
                CustomAttr('combiner', 'i: 0'),
                CustomAttr('T_weight', 'type: DT_RESOURCE')
            ] + string_to_hash.attrs, reshape_2.users)
        fused_ops = [
            cast_1, stride_slice_0, pack_1, cast_0, sparse_reshape_1, slice_0,
            prod_0, gather_v2_2, pack_0, sparse_reshape_0, identity_2,
            identity_1, greater_equal_0, where_0, reshape_0, gather_v2_0,
            gather_v2_1, identity_0, node, unique_0, resource_gather_0,
            identity_3, identity_4, stride_slice_1, cast_2,
            sparse_segment_sum_0, shape_1, stride_slice_2, pack_2, reshape_1,
            tile_0, zeros_like_0, select_0, shape_2, slice_1, cast_3, slice_2,
            concat_v2_0, reshape_2, shape_0, where_1, not_equal,
            string_to_hash, gather_nd, expand_dims
        ]
        replace_ops = [(reshape_2, 0)]
        return custom_node, fused_ops, replace_ops

    def match_seg_mean(self, node: Node):
        self.check_node(node, (OpType.SparseFillEmptyRows, None))
        self.check_operands(node, [(OpType.GatherV2, None),
                                   (OpType.GatherV2, None),
                                   (OpType.Identity, None)])
        gather_v2_0: Node = node.operands[0][0]
        gather_v2_1: Node = node.operands[1][0]
        identity_0: Node = node.operands[2][0]
        self.check_operands(gather_v2_0, [(OpType.SparseReshape, None),
                                          (OpType.Reshape, None)])
        sparse_reshape_0: Node = gather_v2_0.operands[0][0]
        reshape_0: Node = gather_v2_0.operands[1][0]
        self.check_operands(gather_v2_1, [(OpType.Identity, None),
                                          (OpType.Reshape, reshape_0.name)])
        identity_1: Node = gather_v2_1.operands[0][0]
        self.check_operands(identity_0,
                            [(OpType.SparseReshape, sparse_reshape_0.name)])
        self.check_operands(reshape_0, [(OpType.Where, None)])
        where_0: Node = reshape_0.operands[0][0]
        self.check_operands(where_0, [(OpType.GreaterEqual, None)])
        greater_equal_0 = where_0.operands[0][0]
        self.check_operands(greater_equal_0,
                            [(OpType.Identity, identity_1.name)])
        self.check_operands(identity_1,
                            [(OpType.StringToHashBucketFast, None)])
        self.check_operands(sparse_reshape_0, [(OpType.Where, None),
                                               (OpType.Shape, None),
                                               (OpType.Pack, None)])
        where_1: Node = sparse_reshape_0.operands[0][0]
        shape_0: Node = sparse_reshape_0.operands[1][0]
        pack_0: Node = sparse_reshape_0.operands[2][0]
        self.check_operands(pack_0, [(OpType.Prod, None),
                                     (OpType.GatherV2, None)])
        prod_0: Node = pack_0.operands[0][0]
        gather_v2_2: Node = pack_0.operands[1][0]
        self.check_operands(prod_0, [(OpType.Slice, None)])
        slice_0 = prod_0.operands[0][0]
        self.check_operands(slice_0, [(OpType.Shape, shape_0.name)])
        self.check_operands(gather_v2_2, [(OpType.Shape, shape_0.name)])
        self.check_operands(shape_0, [(OpType.ExpandDims, None)])
        expand_dims: Node = shape_0.operands[0][0]
        string_to_hash: Node = identity_1.operands[0][0]
        self.check_operands(string_to_hash, [(OpType.GatherNd, None)])
        gather_nd: Node = string_to_hash.operands[0][0]
        self.check_operands(gather_nd, [(OpType.ExpandDims, expand_dims.name),
                                        (OpType.Where, where_1.name)])
        self.check_operands(where_1, [(OpType.NotEqual, None)])
        not_equal: Node = where_1.operands[0][0]
        self.check_operands(not_equal, [(OpType.ExpandDims, expand_dims.name)])

        self.check_users(node, [(OpType.StridedSlice, None),
                                (OpType.Unique, None), (OpType.Reshape, None)],
                         3)
        stride_slice_1: Node = node.users[0]
        unique_0: Node = node.users[1]
        reshape_1: Node = node.users[2]
        self.check_users(unique_0, [(OpType.GatherV2, None),
                                    (OpType.SparseSegmentMean, None)])
        gather_v2_3: Node = unique_0.users[0]
        sparse_segment_mean_0: Node = unique_0.users[1]
        self.check_users(gather_v2_3, [(OpType.Identity, None)])
        identity_3: Node = gather_v2_3.users[0]
        self.check_users(
            identity_3,
            [(OpType.SparseSegmentMean, sparse_segment_mean_0.name)])
        self.check_users(stride_slice_1, [(OpType.Cast, None)])
        cast_2: Node = stride_slice_1.users[0]
        self.check_users(
            cast_2, [(OpType.SparseSegmentMean, sparse_segment_mean_0.name)])
        self.check_users(sparse_segment_mean_0, [(OpType.Shape, None),
                                                 (OpType.ZerosLike, None),
                                                 (OpType.Select, None)])
        shape_1: Node = sparse_segment_mean_0.users[0]
        zeros_like_0: Node = sparse_segment_mean_0.users[1]
        select_0: Node = sparse_segment_mean_0.users[2]
        self.check_users(shape_1, [(OpType.StridedSlice, None)])
        stride_slice_2: Node = shape_1.users[0]
        self.check_users(stride_slice_2, [(OpType.Pack, None)])
        pack_1: Node = stride_slice_2.users[0]
        self.check_users(pack_1, [(OpType.Tile, None)])
        tile_0: Node = pack_1.users[0]
        self.check_users(reshape_1, [(OpType.Tile, tile_0.name)])
        self.check_users(tile_0, [(OpType.Select, select_0.name)])
        self.check_users(zeros_like_0, [(OpType.Select, select_0.name)])

        self.check_users(select_0, [(OpType.Shape, None),
                                    (OpType.Reshape, None)])
        shape_2: Node = select_0.users[0]
        reshape_2: Node = select_0.users[1]
        self.check_users(shape_2, [(OpType.Slice, None)])
        slice_1: Node = shape_2.users[0]
        self.check_users(slice_1, [(OpType.ConcatV2, None)])
        concat_v2_0: Node = slice_1.users[0]
        self.check_users(shape_0, [(OpType.Slice, None),
                                   (OpType.GatherV2, None),
                                   (OpType.SparseReshape, None),
                                   (OpType.Cast, None)])
        cast_3: Node = shape_0.users[3]
        self.check_users(cast_3, [(OpType.Slice, None)])
        slice_2: Node = cast_3.users[0]
        self.check_users(slice_2, [(OpType.ConcatV2, concat_v2_0.name)])
        self.check_users(concat_v2_0, [(OpType.Reshape, reshape_2.name)])
        self.check_users(reshape_2, [(OpType.Shape, None),
                                     (OpType.Reshape, None)])
        shape_3: Node = reshape_2.users[0]
        reshape_3: Node = reshape_2.users[1]
        self.check_users(shape_3, [(OpType.StridedSlice, None)])
        stride_slice_3: Node = shape_3.users[0]
        self.check_users(stride_slice_3, [(OpType.Pack, None)])
        pack_2: Node = stride_slice_3.users[0]
        self.check_users(pack_2, [(OpType.Reshape, reshape_3.name)])

        # combiner: 0 for sum, 1 for mean
        custom_node = CustomNode(
            'KPFusedEmbeddingWithHashBucket', node.name + '/rec_embed_kp_bucket',
            self.graph, reshape_3.output_shapes,
            expand_dims.operands[:1] + gather_v2_3.operands[:1],
            string_to_hash.attrs + [
                CustomAttr('combiner', 'i: 1'),
                CustomAttr('T_weight', 'type: DT_FLOAT')
            ], reshape_3.users)
        fused_ops = [
            slice_0, prod_0, gather_v2_2, pack_0, sparse_reshape_0, identity_1,
            greater_equal_0, where_0, reshape_0, identity_0, gather_v2_1,
            gather_v2_0, node, unique_0, gather_v2_3, identity_3,
            stride_slice_1, cast_2, sparse_segment_mean_0, shape_1,
            stride_slice_2, pack_1, tile_0, zeros_like_0, select_0, shape_2,
            slice_1, cast_3, slice_2, concat_v2_0, reshape_2, shape_3,
            stride_slice_3, pack_2, reshape_3, reshape_1, string_to_hash,
            gather_nd, where_1, shape_0, not_equal, expand_dims
        ]
        replace_ops = [(reshape_3, 0)]
        return custom_node, fused_ops, replace_ops

    def match_and_rewrite(self, node: Node):
        try:
            res = self.match_seg_sum(node)
        except CheckFailed:
            res = None
        if res is None:
            res = self.match_seg_mean(node)
        custom_node, fused_ops, replace_ops = res

        index = node.get_index()
        self.graph.nodes.insert(index + 1, custom_node)
        for rep_op in replace_ops:
            self.replace_all_users_with(*rep_op, self.graph.nodes[index + 1],
                                        0)
        for fused_op in fused_ops:
            self.graph.delete_node(fused_op)


class LinearSparseEmbeddingPatternRewriter(BaseRewriter):

    def match_and_rewrite(self, node: Node):
        self.check_node(node, (OpType.ExpandDims, None))
        self.check_operands(node, [(OpType.Placeholder, None),
                                   (OpType.Const, None)])
        expand_dim = node.operands[1][0]
        self.check_users(node,
                         [(OpType.NotEqual, None), (OpType.GatherNd, None),
                          (OpType.Shape, None)], 3)
        not_equal, gather_nd, shape_op = node.users
        self.check_users(not_equal, [(OpType.Where, None)], 1)
        where_op = not_equal.users[0]
        self.check_users(where_op, [(OpType.GatherNd, gather_nd.name),
                                    (OpType.SparseReshape, None)])
        sparse_reshape = where_op.users[1]
        self.check_users(gather_nd, [(OpType.StringToHashBucketFast, None)], 1)
        string_to_hash_bucket_fast = gather_nd.users[0]
        self.check_users(string_to_hash_bucket_fast, [(OpType.Identity, None)],
                         1)
        identity = string_to_hash_bucket_fast.users[0]
        self.check_users(identity, [(OpType.Identity, None)], 1)
        identity_2 = identity.users[0]
        self.check_users(identity_2, [(OpType.GreaterEqual, None),
                                      (OpType.GatherV2, None)], 2)
        greater_equal, gather_v2 = identity_2.users
        self.check_users(greater_equal, [(OpType.Where, None)], 1)
        where_op_2 = greater_equal.users[0]
        self.check_users(where_op_2, [(OpType.Reshape, None)], 1)
        reshape = where_op_2.users[0]
        self.check_users(reshape, [(OpType.GatherV2, None),
                                   (OpType.GatherV2, gather_v2.name)], 2)
        gather_v2_1, gather_v2_2 = reshape.users
        self.check_users(shape_op,
                         [(OpType.Cast, None),
                          (OpType.SparseReshape, sparse_reshape.name)], 2)
        cast_op = shape_op.users[0]
        self.check_users(cast_op, [(OpType.StridedSlice, None)], 1)
        strideslice_op = cast_op.users[0]
        self.check_users(strideslice_op, [(OpType.Pack, None)], 1)
        pack_op = strideslice_op.users[0]
        self.check_users(pack_op, [(OpType.Cast, None)], 1)
        cast_op_1 = pack_op.users[0]
        self.check_users(cast_op_1,
                         [(OpType.SparseReshape, sparse_reshape.name)], 1)
        self.check_users(sparse_reshape, [(OpType.Slice, None),
                                          (OpType.GatherV2, None),
                                          (OpType.SparseReshape, None),
                                          (OpType.SparseReshape, None),
                                          (OpType.Cast, None)], 5)
        slice_1, gather_v2, sparse_reshape_2 = sparse_reshape.users[:3]
        self.check_users(slice_1, [(OpType.Prod, None)], 1)
        prod = slice_1.users[0]
        self.check_users(prod, [(OpType.Pack, None)], 1)
        pack_op_1 = prod.users[0]
        self.check_users(gather_v2, [(OpType.Pack, pack_op_1.name)], 1)
        self.check_users(sparse_reshape_2,
                         [(OpType.GatherV2, gather_v2_1.name),
                          (OpType.Identity, None)], 2)
        identity_3 = sparse_reshape_2.users[1]
        self.check_users(identity_3, [(OpType.SparseFillEmptyRows, None)], 1)
        sparse_fill_empty_rows = identity_3.users[0]
        self.check_users(
            gather_v2_1,
            [(OpType.SparseFillEmptyRows, sparse_fill_empty_rows.name)], 1)
        self.check_users(
            gather_v2_2,
            [(OpType.SparseFillEmptyRows, sparse_fill_empty_rows.name)], 1)

        index = node.get_index()
        self.graph.nodes.insert(
            index + 1,
            CustomNode(
                'KPFusedSparseEmbedding',
                sparse_fill_empty_rows.name + '/rec_embed_kp_sparse',
                self.graph, sparse_fill_empty_rows.output_shapes[:3] +
                [sparse_reshape.output_shapes[1]], node.operands[:1],
                string_to_hash_bucket_fast.attrs,
                sparse_fill_empty_rows.users + [sparse_reshape.users[4]]))

        self.replace_all_users_with(sparse_reshape, 1,
                                    self.graph.nodes[index + 1], 3)
        self.replace_all_users_with(sparse_fill_empty_rows, 0,
                                    self.graph.nodes[index + 1], 0)
        self.replace_all_users_with(sparse_fill_empty_rows, 1,
                                    self.graph.nodes[index + 1], 1)
        self.replace_all_users_with(sparse_fill_empty_rows, 2,
                                    self.graph.nodes[index + 1], 2)

        fused_ops = [
            expand_dim, node, not_equal, where_op, gather_nd,
            string_to_hash_bucket_fast, identity, identity_2, greater_equal,
            where_op_2, reshape, gather_v2_1, gather_v2_2, shape_op, cast_op,
            strideslice_op, pack_op, cast_op_1, sparse_reshape, slice_1, prod,
            gather_v2, pack_op_1, sparse_reshape_2, identity_3,
            sparse_fill_empty_rows
        ]
        for fused_op in fused_ops:
            self.graph.delete_node(fused_op)
