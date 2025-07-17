import os
from typing import List
import google.protobuf
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2


def convert_saved_model_to_pbtxt(model_path):
    saved_model = saved_model_pb2.SavedModel()
    with open(os.path.join(model_path, 'saved_model.pb'), 'rb') as f:
        saved_model.ParseFromString(f.read())
    with open(os.path.join(model_path, 'saved_model.pbtxt'), 'w') as f:
        f.write(google.protobuf.text_format.MessageToString(saved_model))


def convert_pbtxt_to_saved_model(model_path):
    saved_model = saved_model_pb2.SavedModel()
    with open(os.path.join(model_path, 'saved_model.pbtxt'), 'r') as f:
        google.protobuf.text_format.Parse(f.read(), saved_model)
    with open(os.path.join(model_path, 'saved_model.pb'), 'wb') as f:
        f.write(saved_model.SerializeToString())
    print('>>', os.path.join(model_path, 'saved_model.pb'))


DTYPE = {
    'string': 'DT_STRING',
    'float32': 'DT_FLOAT',
    'float16': 'DT_HALF',
    'int64': 'DT_INT64',
    'int32': 'DT_INT32',
    'float64': 'DT_DOUBLE',
    'uint32': 'DT_UINT32',
    'uint64': 'DT_UINT64',
    'bool': 'DT_BOOL',
}
ENUM_DTYPE = {
    types_pb2.DT_INT64: ('DT_INT64', 'int64_val'),
    types_pb2.DT_FLOAT: ('DT_FLOAT', 'float_val'),
    types_pb2.DT_HALF: ('DT_HALF', 'half_val'),
    types_pb2.DT_DOUBLE: ('DT_DOUBLE', 'double_val'),
    types_pb2.DT_INT32: ('DT_INT32', 'int_val'),
    types_pb2.DT_UINT32: ('DT_UINT32', 'uint32_val'),
    types_pb2.DT_UINT64: ('DT_UINT64', 'uint64_val'),
    types_pb2.DT_STRING: ('DT_STRING', 'string_val'),
    types_pb2.DT_INT64: ('DT_INT64', 'int64_val'),
    types_pb2.DT_BOOL: ('DT_BOOL', 'bool_val'),
}


class Attr:

    def __init__(self, key, value):
        self.key: str = key
        self.value = value


class CustomAttr:

    def __init__(self, key, value):
        self.key: str = key
        self.value = value


class Node:

    def __init__(self, node: node_def_pb2.NodeDef, graph):
        self.node = node
        self.graph = graph
        self.type = node.op
        self.name = node.name
        self.operands: List[tuple] = []
        self.users: List[Node] = []
        self.attrs: List[Attr] = []
        self.load_attrs(node)

        for i in self.load_inputs(node.input):
            i_name = i.split(':')[0]
            if i_name in self.graph.tensors:
                self.graph.tensors[i_name].users.append(self)

    def get_index(self) -> int:
        for i, node in enumerate(self.graph.nodes):
            if node.name == self.name:
                return i
        raise ValueError(f'Node "{self.name}" not in graph')

    def load_inputs(self, inputs):
        input_tensors = []
        for i_name in inputs:
            input_tensors.append(i_name)
            name = i_name.split(':')[0]
            index = 0 if ':' not in i_name else i_name.split(':')[1]
            operand = self.graph.tensors[name] if name in self.graph.tensors else None
            self.operands.append((operand, int(index)))
        return input_tensors

    def load_attrs(self, node: node_def_pb2.NodeDef):
        for key, attr in node.attr.items():
            self.attrs.append(Attr(key, attr))

    def as_node_def(self):
        inputs: List[str] = []
        for operand, index in self.operands:
            if index == 0:
                inputs.append(operand.name)
            else:
                inputs.append(f'{operand.name}:{index}')
        return node_def_pb2.NodeDef(
            name=self.name,
            op=self.type,
            input=inputs,
            attr={attr.key: attr.value for attr in self.attrs}
        )


class CustomNode:

    def __init__(self, type: str, name: str, graph, outputs: List[tuple],
                 operands: List[Node], attrs: List[Attr], users: List[Node]):
        self.type = type
        self.name = name
        self.graph = graph
        self.output_shapes = outputs
        self.attrs = attrs
        self.operands = operands
        self.users = users
        self.control_inputs = []
        self.device = None
    
    def get_index(self) -> int:
        for i, node in enumerate(self.graph.nodes):
            if node.name == self.name:
                return i
        raise ValueError(f'Node "{self.name}" not in graph')

    def as_node_def(self):
        inputs: List[str] = []
        for operand, index in self.operands:
            if index == 0:
                inputs.append(operand.name)
            else:
                inputs.append(f'{operand.name}:{index}')
        return node_def_pb2.NodeDef(
            name=self.name,
            op=self.type,
            input=inputs,
            attr={attr.key: attr.value for attr in self.attrs}
        )


class Graph:

    def __init__(self):
        self.versions = {'producer': None}
        self.nodes: List[Node] = []
        self.tensors = {}

    def load_graph(self, graph_def: graph_pb2.GraphDef):
        self.graph_def = graph_def
        for node in graph_def.node:
            self.nodes.append(Node(node, self))
            self.tensors[self.nodes[-1].name] = self.nodes[-1]

    def get_node(self, name: str) -> Node:
        for node in self.nodes:
            if node.name == name:
                return node
        raise ValueError(f'Node "{name}" not found')

    def insert_before_node(self, insert_node: CustomNode, 
                           before_node: Node,
                           graph_nodes: List[node_def_pb2.NodeDef]):
        index = -1
        for i, node in enumerate(graph_nodes):
            if node.name == before_node.name:
                index = i
                break
        if index == -1:
            raise ValueError(f'Node "{before_node.name}" not in graph')
        graph_nodes.insert(index, insert_node.as_node_def())
        
        self.nodes.insert(before_node.get_index(), insert_node)
        # print(f'>> add node: [{insert_node.type}] \'{insert_node.name}\'')

    def delete_nodes(self, delete_nodes: List[Node], 
                     graph_nodes: List[node_def_pb2.NodeDef]):
        names: List[str] = [node.name for node in delete_nodes]
        indexes: List[int] = []
        for i, node in enumerate(graph_nodes):
            if node.name in names:
                indexes.append(i)
        for index in sorted(indexes, reverse=True):
            # print(f'-- delete node: [{graph_nodes[index].op}] \'{graph_nodes[index].name}\'')
            graph_nodes.pop(index)
        
        indexes: List[int] = []
        for i, n in enumerate(self.nodes):
            if n.name in names:
                indexes.append(i)
        for i in sorted(indexes, reverse=True):
            # print(f'-- delete op: [{self.nodes[i].type}] \'{self.nodes[i].name}\'')
            del self.nodes[i]

    def delete_node(self, node: Node, graph_nodes: List[node_def_pb2.NodeDef]):
        for i, n in enumerate(graph_nodes):
            if n.name == node.name:
                print(f'-- delete node: [{n.op}] \'{n.name}\'')
                graph_nodes.pop(i)
                break
        for i, n in enumerate(self.nodes):
            if n.name == node.name:
                del self.nodes[i]
                break

    def replace_all_users_with(self, old_node: Node, old_index: int,
                               new_node: Node, new_index: int, 
                               graph_nodes: List[node_def_pb2.NodeDef]):
        for user in old_node.users:
            for i, operand in enumerate(user.operands):
                if operand[0].name == old_node.name and \
                    operand[1] == old_index:
                    user.operands[i] = (new_node, new_index)
        
        old_name = old_node.name if old_index == 0 else f'{old_node.name}:{old_index}'
        new_name = new_node.name if new_index == 0 else f'{new_node.name}:{new_index}'
        for i, node in enumerate(graph_nodes):
            for j, i_name in enumerate(node.input):
                if i_name == old_name:
                    graph_nodes[i].input[j] = new_name


class MetaGraph:

    def __init__(self, model_path: str):
        self.graph = Graph()
        self.model_path = model_path
        self.load()

    def load(self):
        self.saved_model = saved_model_pb2.SavedModel()
        pb_file = os.path.join(self.model_path, 'saved_model.pb')
        pbtxt_file = os.path.join(self.model_path, 'saved_model.pbtxt')
        if os.path.exists(pb_file):
            with open(pb_file, 'rb') as f:
                self.saved_model.ParseFromString(f.read())
        elif os.path.exists(pbtxt_file):
            with open(pbtxt_file, 'r') as f:
                google.protobuf.text_format.Parse(f.read(), self.saved_model)
        self.graph.load_graph(self.saved_model.meta_graphs[0].graph_def)

    def save(self, output_path, to_text: bool = False):
        os.makedirs(output_path, exist_ok=True)
        if to_text:
            self.write_code(self.saved_model, output_path)
        else:
            # default to binary
            self.write_binary(self.saved_model, output_path)
    
    def write_binary(self, model: saved_model_pb2.SavedModel, output_path: str):
        output_file = os.path.join(output_path, 'saved_model.pb')
        if os.path.exists(output_file):
            overwrite = input('saved_model.pb is already exist, '
                              'do you want to overwrite it? (y/n) ')
            if overwrite.lower() != 'y':
                print('Aborted')
                return
        with tf.io.gfile.GFile(output_file, 'wb') as f:
            f.write(model.SerializeToString())
        print('>>', output_file)
    
    def write_code(self, model: saved_model_pb2.SavedModel, output_path: str):
        output_file = os.path.join(output_path, 'saved_model.pbtxt')
        if os.path.exists(output_file):
            overwrite = input('saved_model.pbtxt is already exist, '
                              'do you want to overwrite it? (y/n) ')
            if overwrite.lower() != 'y':
                print('Aborted')
                return
        with tf.io.gfile.GFile(output_file, 'w') as f:
            f.write(str(model))
        print('>>', output_file)
