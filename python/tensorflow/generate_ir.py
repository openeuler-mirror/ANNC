import os
from typing import List
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf.saved_model_pb2 import SavedModel
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import function_pb2

class SavedModelParser:
    def __init__(self):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.model = SavedModel()
    
    def parse(self, model_path: str):
        assert os.path.exists(model_path)
        with tf.io.gfile.GFile(os.path.join(model_path, 'saved_model.pb'), 'rb') as f:
            self.model.ParseFromString(f.read())
        return self.build()

    def build(self):
        graph_def = self.model.meta_graphs[0].graph_def
        self.graph = self.build_graph(graph_def)
        func_idx = 0
        for lib_func in graph_def.library.function:
            self.build_function(lib_func, func_idx)
            func_idx += 1
        return self.graph
    
    def build_graph(self, graph_def: graph_pb2.GraphDef):
        nodes_name = {}
        node_info_list = []
        for i, node in enumerate(graph_def.node):
            attrs = []
            for attr_name, attr_value in node.attr.items():
                if attr_name == 'value':
                    attr_value.tensor.tensor_content = b''
                attr_value_str = str(attr_value).replace('\n', ' ').replace(' ','')
                attr_str = f"{attr_name}: {attr_value_str}"
                attrs.append(attr_str)
            attrs_str = ", ".join(attrs)
            
            node_operand = []
            if node.input:
                for input_name in node.input:
                    node_operand.append(nodes_name.get(input_name.split(":")[0], f"%{input_name}"))
            
            node_str = f"%{node.op}.{i}"
    
            node_operand_str = ", ".join(node_operand)
            
            nodes_name[node.name] = node_str
            
            node_info = f'{node_str} = %{node.op}({node_operand_str}) {{op_name="{node.name}"}} {{op_attrs={attrs_str}}}'
            node_info_list.append(node_info)
        print("\nGraph Node Information:")
        print("{")
        for info in node_info_list:
            print(info)
        print("}")
        
    def build_function(self, lib_func: function_pb2.FunctionDef, func_idx: int):
        nodes_name = {}
        node_info_list = []
        for i, node in enumerate(lib_func.node_def):
            attrs = []
            for attr_name, attr_value in node.attr.items():
                if attr_name == 'value':
                    attr_value.tensor.tensor_content = b''
                attr_value_str = str(attr_value).replace('\n', ' ').replace(' ','')
                attr_str = f"{attr_name}: {attr_value_str}"
                attrs.append(attr_str)
            attrs_str = ", ".join(attrs)
            
            node_operand = []
            if node.input:
                for input_name in node.input:
                    node_operand.append(nodes_name.get(input_name.split(":")[0], f"%{input_name}"))
            
            node_str = f"%{node.op}.{i}"
    
            node_operand_str = ", ".join(node_operand)
            
            nodes_name[node.name] = node_str
            
            node_info = f'{node_str} = %{node.op}({node_operand_str}) {{op_name="{node.name}"}} {{op_attrs={attrs_str}}}'
            node_info_list.append(node_info)
        print(f"\nFunction %{func_idx} Node Information:")
        print("{")
        for info in node_info_list:
            print(info)
        print("}")
            
if __name__ == '__main__':
    import sys
    # path to model.pb
    input_path = sys.argv[1]
    # save model to output_path
    # output_path = sys.argv[2]

    # Parse the model
    parser = SavedModelParser()    
    g = parser.parse(input_path)
