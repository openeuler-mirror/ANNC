from .graph import Node, CustomNode, CustomAttr
from .rewriter import BaseRewriter, CheckFailed
from .op_type import OpType
from tensorflow.core.framework import attr_value_pb2


class KPSparseSelectPatternRewriter(BaseRewriter):
    __pattern__ = 'KPFusedSparseSelect'

    def match_and_rewrite(self, node: Node):
        self.check_node(node, (OpType.Concat, None))
        self.check_operands(node, [(OpType.Mul, None),
                                   (OpType.Select, None),
                                   (OpType.Const, None)])
        mul_op: Node = node.operands[0][0]
        select_op: Node = node.operands[1][0]
        self.check_operands(mul_op, [(OpType.Const, None),
                                     (OpType.Select, None)])
        select_1_op: Node = mul_op.operands[1][0]
        self.check_operands(select_1_op, [(OpType.Equal, None),
                                          (OpType.Fill, None),
                                          (OpType.Select, None)])
        equal_op: Node = select_1_op.operands[0][0]
        fill_op: Node = select_1_op.operands[1][0]
        select_2_op: Node = select_1_op.operands[2][0]
        self.check_operands(equal_op, [(OpType.Reshape, None),
                                       (OpType.Const, None)])
        reshape_op: Node = equal_op.operands[0][0]
        self.check_operands(select_2_op, [(OpType.Equal, None),
                                          (OpType.Fill, fill_op.name),
                                          (OpType.Cast, None)])
        equal_1_op: Node = select_2_op.operands[0][0]
        cast_op: Node = select_2_op.operands[2][0]
        self.check_operands(equal_1_op, [(OpType.Reshape, reshape_op.name),
                                         (OpType.Const, None)])
        self.check_operands(fill_op, [(OpType.Shape, None),
                                      (OpType.Const, None)])
        shape_op: Node = fill_op.operands[0][0]
        self.check_operands(shape_op, [(OpType.Cast, cast_op.name)])
        self.check_operands(cast_op, [(OpType.Greater, None)])
        greater_op: Node = cast_op.operands[0][0]
        self.check_operands(greater_op, [(OpType.Reshape, None),
                                         (OpType.Const, None)])
        reshape_1_op: Node = greater_op.operands[0][0]
        self.check_operands(select_op, [(OpType.Equal, None),
                                        (OpType.RealDiv, None),
                                        (OpType.Fill, None)])
        equal_2_op: Node = select_op.operands[0][0]
        realdiv_op: Node = select_op.operands[1][0]
        fill_1_op: Node = select_op.operands[2][0]
        self.check_operands(equal_2_op, [(OpType.Reshape, None),
                                         (OpType.Const, None)])
        reshape_2_op: Node = equal_2_op.operands[0][0]
        self.check_operands(realdiv_op, [(OpType.Fill, fill_1_op.name),
                                         (OpType.Const, None)])
        self.check_operands(fill_1_op, [(OpType.Shape, None),
                                        (OpType.Const, None)])
        shape_1_op: Node = fill_1_op.operands[0][0]
        self.check_operands(shape_1_op, [(OpType.Reshape, reshape_1_op.name)])
        
        print(f'>> Add fusion [{self.__pattern__}]:', node.name)

        output_shapes = attr_value_pb2.AttrValue()
        for attr in reshape_1_op.attrs + select_1_op.attrs + node.attrs:
            if attr.key == '_output_shapes':
                output_shapes.list.shape.extend(attr.value.list.shape)
        
        index = reshape_1_op.get_index()
        
        insert_node = CustomNode(
            self.__pattern__,
            node.name + '/kp_fused',
            self.graph, 
            [], 
            [reshape_1_op.operands[0], reshape_op.operands[0], reshape_2_op.operands[0]],
            [CustomAttr('_output_shapes', output_shapes)],
            reshape_1_op.users + select_1_op.users + node.users)
        self.graph.insert_before_node(insert_node, before_node=reshape_1_op, 
                                      graph_nodes=self.graph.graph_def.node)
        self.graph.replace_all_users_with(reshape_1_op, 0, self.graph.nodes[index], 0,
                                          graph_nodes=self.graph.graph_def.node)
        self.graph.replace_all_users_with(select_1_op, 0, self.graph.nodes[index], 1,
                                          graph_nodes=self.graph.graph_def.node)
        self.graph.replace_all_users_with(node, 0, self.graph.nodes[index], 2,
                                          graph_nodes=self.graph.graph_def.node)

        self.graph.delete_nodes([node, mul_op, select_op, select_1_op, equal_op, fill_op,
                                 select_2_op, reshape_op, equal_1_op, cast_op, shape_op,
                                 greater_op, reshape_1_op, equal_2_op, realdiv_op, fill_1_op,
                                 reshape_2_op, shape_1_op], graph_nodes=self.graph.graph_def.node)
        return insert_node.get_index()


class KPFusedGatherPatternRewriter(BaseRewriter):
    __pattern__ = 'KPFusedGather'

    def match_and_rewrite(self, node: Node):
        self.check_node(node, (OpType.GatherV2, None))
        self.check_operands(node, [(OpType.Identity, None),
                                   (OpType.Unique, None),
                                   (OpType.Const, None)])
        identity_op: Node = node.operands[0][0]
        unique_op: Node = node.operands[1][0]
        self.check_operands(identity_op, [(OpType.GatherV2, None)])
        gatherv2_op: Node = identity_op.operands[0][0]
        self.check_operands(gatherv2_op, [(OpType.Identity, None),
                                          (OpType.Unique, unique_op.name),
                                          (OpType.Const, None)])
        self.check_operands(unique_op, [(OpType.Reshape, None)])
        reshape_op: Node = unique_op.operands[0][0]
        self.check_operands(reshape_op, [(OpType.Unique, None),
                                         (OpType.Const, None)])
        unique_1_op: Node = reshape_op.operands[0][0]
        shape_op: Node = None
        for i, n in enumerate(unique_1_op.users):
            if n.type == OpType.Shape:
                shape_op = unique_1_op.users[i]
        if shape_op is None:
            raise CheckFailed
        self.check_operands(unique_1_op, [(OpType.Identity, None)])
        identity_1_op: Node = unique_1_op.operands[0][0]
        self.check_operands(identity_1_op, [(OpType.StridedSlice, None)])
        strided_slice: Node = identity_1_op.operands[0][0]
        
        print(f'>> Add fusion [{self.__pattern__}]:', unique_1_op.name)

        output_shapes = attr_value_pb2.AttrValue()
        for attr in [shape_op, node.attrs]:
            if attr.key == '_output_shapes':
                output_shapes.list.shape.extend(attr.value.list.shape)
        
        index = unique_1_op.get_index()
        
        insert_node = CustomNode(
            self.__pattern__,
            node.name + '/kp_fused',
            self.graph, 
            [], 
            [gatherv2_op.operands[0]] + strided_slice.operands[:2],
            [CustomAttr('_output_shapes', output_shapes)],
            shape_op.users, unique_1_op.users + node.users)
        self.graph.insert_before_node(insert_node, before_node=unique_1_op, 
                                      graph_nodes=self.graph.graph_def.node)
        self.graph.replace_all_users_with(shape_op, 0, self.graph.nodes[index], 0,
                                          graph_nodes=self.graph.graph_def.node)
        self.graph.replace_all_users_with(unique_1_op, 0, self.graph.nodes[index], 1,
                                          graph_nodes=self.graph.graph_def.node)
        self.graph.replace_all_users_with(node, 0, self.graph.nodes[index], 2,
                                          graph_nodes=self.graph.graph_def.node)

        self.graph.delete_nodes([node, identity_op, unique_op, gatherv2_op, reshape_op,
                                 unique_1_op, shape_op, identity_1_op, strided_slice
                                 ], graph_nodes=self.graph.graph_def.node)
        return insert_node.get_index()


class KPSparseReshapePatternRewriter(BaseRewriter):
    __pattern__ = 'KPFusedSparseReshape'

    def match_and_rewrite(self, node: Node):
        self.check_node(node, (OpType.SparseReshape, None))
        self.check_operands(node, [(OpType.Concat, None),
                                   (OpType.Cast, None),
                                   (OpType.Cast, None)])
        concat_op: Node = node.operands[0][0]
        cast_op: Node = node.operands[1][0]
        cast_1_op: Node = node.operands[2][0]
        self.check_operands(cast_1_op, [(OpType.Const, None)])
        self.check_operands(cast_op, [(OpType.Pack, None)])
        pack_op: Node = cast_op.operands[0][0]
        self.check_operands(pack_op, [(OpType.StridedSlice, None),
                                      (OpType.Const, None)])
        strided_slice: Node = pack_op.operands[0][0]
        self.check_operands(strided_slice, [(OpType.Shape, None),
                                            (OpType.Const, None),
                                            (OpType.Const, None)
                                            (OpType.Const, None)])
        shape_op: Node = strided_slice.operands[0][0]
        self.check_operands(concat_op, [(OpType.Cast, None),
                                        (OpType.Reshape, None),
                                        (OpType.Const, None)])
        cast_2_op: Node = concat_op.operands[0][0]
        reshape_op: Node = concat_op.operands[1][0]
        self.check_operands(reshape_op, [(OpType.StridedSlice, None),
                                         (OpType.Const, None)])
        strided_slice_1: Node = reshape_op.operands[0][0]
        self.check_operands(cast_2_op, [(OpType.Reshape, None)])
        reshape_1_op: Node = cast_2_op.operands[0][0]
        self.check_operands(reshape_1_op, [(OpType.Range, None),
                                           (OpType.Const, None)])
        range_op: Node = reshape_1_op.operands[0][0]
        self.check_operands(range_op, [(OpType.Const, None),
                                       (OpType.StridedSlice, strided_slice.name), 
                                       (OpType.Const, None)])
        
        print(f'>> Add fusion [{self.__pattern__}]:', node.name)

        output_shapes = attr_value_pb2.AttrValue()
        for attr in node.attrs:
            if attr.key == '_output_shapes':
                output_shapes.list.shape.extend(attr.value.list.shape)
        
        index = node.get_index()
        
        insert_node = CustomNode(
            self.__pattern__,
            node.name + '/kp_fused',
            self.graph, 
            [], 
            [shape_op.operands[0], strided_slice_1.operands[1], cast_1_op.operands[0]],
            [CustomAttr('_output_shapes', output_shapes)],
            node.users)
        self.graph.insert_before_node(insert_node, before_node=node, 
                                      graph_nodes=self.graph.graph_def.node)
        self.graph.replace_all_users_with(node, 0, self.graph.nodes[index], 0,
                                          graph_nodes=self.graph.graph_def.node)

        self.graph.delete_nodes([node, cast_op, concat_op, cast_1_op, pack_op, strided_slice,
                                 shape_op, cast_2_op, reshape_op, strided_slice_1,
                                 reshape_1_op, range_op], graph_nodes=self.graph.graph_def.node)
        return insert_node.get_index()


class KPEmbeddingActionIdGatherPatternRewriter(BaseRewriter):
    __pattern__ = 'KPFusedEmbeddingActionIdGather'

    def match_and_rewrite(self, node: Node):
        self.check_node(node, (OpType.ConcatV2, None))
        self.check_operands(node, [(OpType.Reshape, None),
                                   (OpType.Fill, None),
                                   (OpType.Const, None)])
        reshape_op: Node = node.operands[0][0]
        fill_op: Node = node.operands[1][0]
        self.check_operands(fill_op, [(OpType.Pack, None),
                                      (OpType.Const, None)])
        pack_op: Node = fill_op.operands[0][0]
        self.check_operands(pack_op, [(OpType.StridedSlice, None),
                                      (OpType.Const, None)])
        self.check_operands(reshape_op, [(OpType.Identity, None),
                                         (OpType.Pack, pack_op.name)])
        identity_op: Node = reshape_op.operands[0][0]
        self.check_operands(identity_op, [(OpType.GatherV2, None)])
        gatherv2_op: Node = identity_op.operands[0][0]
        self.check_operands(gatherv2_op, [(OpType.Identity, None),
                                          (OpType.Sub, None),
                                          (OpType.Const, None)])
        identity_1_op: Node = gatherv2_op.operands[0][0]
        self.check_operands(identity_1_op, [(OpType.GatherV2, None)])
        gatherv2_1_op: Node = identity_1_op.operands[0][0]
        
        print(f'>> Add fusion [{self.__pattern__}]:', node.name)

        output_shapes = attr_value_pb2.AttrValue()
        for attr in node.attrs:
            if attr.key == '_output_shapes':
                output_shapes.list.shape.extend(attr.value.list.shape)
        
        index = node.get_index()
        
        insert_node = CustomNode(
            self.__pattern__,
            node.name + '/kp_fused',
            self.graph, 
            [], 
            [gatherv2_1_op.operands[1], gatherv2_1_op.operands[0], 
             gatherv2_op.operands[1], pack_op.operands[0]],
            [CustomAttr('_output_shapes', output_shapes)],
            node.users)
        self.graph.insert_before_node(insert_node, before_node=node, 
                                      graph_nodes=self.graph.graph_def.node)
        self.graph.replace_all_users_with(node, 0, self.graph.nodes[index], 0,
                                          graph_nodes=self.graph.graph_def.node)

        self.graph.delete_nodes([node, reshape_op, fill_op, pack_op, identity_op, 
                                 gatherv2_op, identity_1_op, gatherv2_1_op], 
                                graph_nodes=self.graph.graph_def.node)
        return insert_node.get_index()

