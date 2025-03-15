from .graph import Node, CustomNode
from .rewriter import BaseRewriter
from .op_type import OpType


class EmbeddingV1PatternRewriter(BaseRewriter):

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
                'RecEmbeddingV1',
                sparse_fill_empty_rows.name + '/rec_embed_v1', self.graph,
                sparse_fill_empty_rows.output_shapes[:3] +
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


class EmbeddingV2PatternRewriter(BaseRewriter):

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
            CustomNode('RecEmbeddingV2',
                       sparse_fill_empty_rows.name + '/rec_embed_v2',
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
