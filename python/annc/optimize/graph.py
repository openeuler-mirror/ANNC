import os
from typing import List
import google.protobuf
import tensorflow as tf
from tensorflow.python.framework import func_graph
from tensorflow.python.framework.ops import Operation, SymbolicTensor
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.framework import types_pb2


def convert_saved_model_to_pbtxt(model_path):
    saved_model = saved_model_pb2.SavedModel()
    with open(os.path.join(model_path, 'saved_model.pb'), 'rb') as f:
        saved_model.ParseFromString(f.read())
    with open(os.path.join(model_path, 'saved_model.pbtxt'), 'w') as f:
        f.write(google.protobuf.text_format.MessageToString(saved_model))


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

    def __init__(self, key, attr):
        self.key: str = key
        self.attr = attr
        self.type = attr.type

    def __str__(self) -> str:
        attr_str = 'attr {\n'
        attr_str += f'  key: "{self.key}"\n'
        attr_str += '  value {\n'
        attr_str += '\n'.join([
            '    ' + li for li in self.attr.__str__().split('\n')
            if li.strip()
        ])
        attr_str += '\n  }\n}\n'
        return attr_str


class CustomAttr:

    def __init__(self, key, value):
        self.key: str = key
        self.value = value

    def __str__(self) -> str:
        return f'''attr {{
  key: "{self.key}"
  value {{
    {self.value}
  }}
}}\n'''


class Node:

    def __init__(self, op: Operation, graph):
        self.node = op
        self.graph = graph
        self.type = op.type
        self.name = op.name
        self.operands: List[tuple] = []
        self.users: List[Node] = []
        self.output_shapes = self.load_outputs(op.outputs)
        self.attrs: List[Attr] = self.load_attrs(op)

        for i in self.load_inputs(op.inputs):
            i_name = i.split(':')[0]
            self.graph.tensors[i_name].users.append(self)

    def get_index(self) -> int:
        for i, node in enumerate(self.graph.nodes):
            if node.name == self.name:
                return i
        raise ValueError(f'Node "{self.name}" not in graph')

    def load_inputs(self, inputs):
        input_tensors = []
        for input in inputs:
            i_name = self.load_input(input)
            input_tensors.append(i_name)
            name, index = i_name.split(':')
            self.operands.append((self.graph.tensors[name], int(index)))
        return input_tensors

    def load_outputs(self, outputs):
        return [(output.shape.rank, output.shape.dims) for output in outputs]

    def load_input(self, tensor: SymbolicTensor):
        return tensor.name

    def load_attrs(self, op: Operation):
        attrs = []
        for key, attr in op.node_def.attr.items():
            attrs.append(Attr(key, attr))
        return attrs

    def __str__(self, indent: int = 6) -> str:

        def get_attr(k, a):
            attr_str = 'attr {\n'
            attr_str += f'  key: "{k}"\n'
            attr_str += '  value {\n'
            attr_str += '\n'.join(
                ['    ' + li for li in a.__str__().split('\n') if li.strip()])
            attr_str += '\n  }\n}\n'
            return attr_str

        attrs = {}
        for key, attr in self.node.node_def.attr.items():
            attrs[key] = get_attr(key, attr).replace('?', '-1')

        if self.type not in ('NoOp', 'AssignVariableOp'):
            output_shapes = 'list {\n'
            for output_shape in self.output_shapes:
                output_shapes += '  shape {\n'
                rank, dims = output_shape
                if rank is None:
                    output_shapes += '    unknown_rank: true\n'
                else:
                    for dim in dims:
                        output_shapes += f'    dim {{\n      size: {dim}\n    }}\n'
                output_shapes += '  }\n'
            output_shapes += '}'
            output_shapes = output_shapes.replace('?', '-1')
            attrs['_output_shapes'] = 'attr {\n'
            attrs['_output_shapes'] += '  key: "_output_shapes"\n'
            attrs['_output_shapes'] += '  value {\n'
            attrs['_output_shapes'] += '\n'.join([
                '    ' + li for li in output_shapes.split('\n') if li.strip()
            ])
            attrs['_output_shapes'] += '\n  }\n}\n'

        node_str = f'name: "{self.name}"\n'
        node_str += f'op: "{self.type}"\n'
        for operand, index in self.operands:
            if index == 0:
                node_str += f'input: "{operand.name}"\n'
            else:
                node_str += f'input: "{operand.name}:{index}"\n'
        for i in self.node.control_inputs:
            node_str += f'input: "^{i.name}"\n'

        if self.node.device:
            node_str += f'device: "{self.node.device}"\n'

        for key, attr in sorted(attrs.items(), key=lambda item: item[0]):
            node_str += attr
        return '\n'.join(
            [indent * ' ' + li for li in node_str.split('\n') if li.strip()])


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

    def __str__(self, indent: int = 6) -> str:
        attrs = {
            attr.key: attr.__str__().replace('?', '-1')
            for attr in self.attrs
        }

        output_shapes = 'list {\n'
        for output_shape in self.output_shapes:
            output_shapes += '  shape {\n'
            rank, dims = output_shape
            if rank is None:
                output_shapes += '    unknown_rank: true\n'
            else:
                for dim in dims:
                    output_shapes += f'    dim {{\n      size: {dim}\n    }}\n'
            output_shapes += '  }\n'
        output_shapes += '}'
        output_shapes = output_shapes.replace('?', '-1')
        attrs['_output_shapes'] = 'attr {\n'
        attrs['_output_shapes'] += '  key: "_output_shapes"\n'
        attrs['_output_shapes'] += '  value {\n'
        attrs['_output_shapes'] += '\n'.join(
            ['    ' + li for li in output_shapes.split('\n') if li.strip()])
        attrs['_output_shapes'] += '\n  }\n}\n'

        node_str = f'name: "{self.name}"\n'
        node_str += f'op: "{self.type}"\n'
        for operand, index in self.operands:
            if index == 0:
                node_str += f'input: "{operand.name}"\n'
            else:
                node_str += f'input: "{operand.name}:{index}"\n'

        for name in self.control_inputs:
            node_str += f'input: "^{name}"\n'
        if self.device:
            node_str += f'device: "{self.device}"\n'

        for key, attr in sorted(attrs.items(), key=lambda item: item[0]):
            node_str += attr
        return '\n'.join(
            [indent * ' ' + li for li in node_str.split('\n') if li.strip()])


class Graph:

    def __init__(self):
        self.versions = {'producer': None}
        self.nodes: List[Node] = []
        self.tensors = {}

    def load_graph(self, graph: func_graph.FuncGraph):
        graph_def = graph.as_graph_def()
        for op in graph.operations:
            self.nodes.append(Node(op, self))
            self.tensors[self.nodes[-1].name] = self.nodes[-1]

        self.versions['producer'] = graph_def.versions.producer

    def get_node(self, name: str) -> Node:
        for node in self.nodes:
            if node.name == name:
                return node
        raise ValueError(f'Node "{name}" not found')

    def delete_node(self, node: Node):
        for i, n in enumerate(self.nodes):
            if n.name == node.name:
                del self.nodes[i]
                return
        raise ValueError(f'Node "{node.name}" not found')

    def dump_graph(self, output_path=None):
        node_defs = []
        for node in self.nodes:
            if node.name == 'dummy_fetch':
                continue
            node_defs.append(f'    node {{\n{node.__str__()}\n    }}')
        node_defs.append('    versions {\n' + \
                         f'      producer: {self.versions["producer"]}' + \
                         '\n    }')
        graph_str = '  graph_def {\n' + '\n'.join(node_defs) + '\n  }'
        if output_path:
            with open(output_path, 'w') as f:
                f.write(graph_str)
        else:
            return graph_str


class MetaGraph:

    def __init__(self, model_path: str):
        self.graph = Graph()
        self.model_path = model_path
        self.load()

    def load(self):
        model = tf.saved_model.load(self.model_path)
        self.graph.load_graph(model.graph)

    def save(self, output_path):
        output_file = os.path.join(output_path, 'saved_model.pbtxt')
        if os.path.exists(output_file):
            overwrite = input('saved_model.pbtxt is already exist, '
                              'do you want to overwrite it? (y/n) ')
            if overwrite.lower() != 'y':
                print('Aborted')
                return

        if os.path.exists(os.path.join(self.model_path, 'saved_model.pbtxt')):
            with open(os.path.join(self.model_path, 'saved_model.pbtxt'),
                      'r') as f:
                text_model = f.read()
        else:
            saved_model = saved_model_pb2.SavedModel()
            with open(os.path.join(self.model_path, 'saved_model.pb'),
                      'rb') as f:
                saved_model.ParseFromString(f.read())
            text_model = google.protobuf.text_format.MessageToString(
                saved_model)

        import re
        graph_def = re.search('  graph_def[\s\S]+?\n  }', text_model).group()
        new_text_model = text_model.replace(graph_def, self.graph.dump_graph())
        os.makedirs(output_path, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(new_text_model)
        print('>>', output_file)
