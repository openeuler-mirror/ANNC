import os
import shutil
from typing import List, Dict
import numpy as np
import tensorflow as tf
from annc.optimize.graph import Graph, CustomAttr
from annc.optimize.op_type import OpType
from annc.optimize.data_pack import DataFormat, MatrixTiling, FormageChange


class LayoutMatmulRewriter:
    __vector_len__ = 12

    def __init__(self, model_path: str):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.model_path = model_path
        self.sess = tf.compat.v1.Session(graph=tf.Graph())
        with self.sess.graph.as_default(), self.sess.as_default():
            tf.compat.v1.saved_model.loader.load(self.sess, ["serve"], model_path)
            self.vars = self.get_variables()

    def __call__(self, graph: Graph):
        with self.sess.graph.as_default(), self.sess.as_default():
            for node in graph.nodes:
                if node.type not in (OpType.MatMul.value,
                                     OpType.BatchMatMul.value):
                    continue

                lhs = self.get_variable_operand(node.operands[0][0])
                if lhs and self.check_require_folding(node, 'lhs_format'):
                    if self.check_data_size(lhs):
                        continue
                    self.pack_variable(lhs, DataFormat.mk, DataFormat.k4m4)
                    node.attrs.append(CustomAttr('lhs_format', 's: "k4m4"'))

                rhs = self.get_variable_operand(node.operands[1][0])
                if rhs and self.check_require_folding(node, 'rhs_format'):
                    if self.check_data_size(rhs):
                        continue
                    self.pack_variable(rhs, DataFormat.kn, DataFormat.kn4)
                    node.attrs.append(CustomAttr('rhs_format', 's: "kn4"'))

    def save(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)

        src_pb = os.path.join(self.model_path, 'saved_model.pb')
        dst_pb = os.path.join(output_path, 'saved_model.pb')
        if os.path.exists(src_pb):
            shutil.copy(src_pb, dst_pb)

        var_path = os.path.join(output_path, 'variables/variables')
        var_dir = os.path.dirname(var_path)
        if os.path.exists(var_dir):
            shutil.rmtree(var_dir)
        os.makedirs(var_dir, exist_ok=True)
        with self.sess.graph.as_default(), self.sess.as_default():
            saver = tf.compat.v1.train.Saver()
            saver.save(self.sess, var_path)
        self.sess.close()

    def check_data_size(self, name: str):
        data: np.ndarray = self.vars[name].eval()
        return (data.shape[0] < self.__vector_len__) or (data.shape[1] < self.__vector_len__)

    def get_variable_operand(self, node) -> str:
        valid_op_types = frozenset(
            [OpType.Identity.value, OpType.ReadVariableOp.value])
        valid_var_types = frozenset([
            OpType.VarHandleOp.value, OpType.VariableV2.value,
            OpType.Variable.value
        ])
        if node.type in valid_op_types:
            return self.get_variable_operand(node.operands[0][0])
        if node.type in valid_var_types:
            return node.name

    def pack_variable(self, name: str, src_format: DataFormat,
                      dst_format: DataFormat):
        data: np.ndarray = self.vars[name].eval()
        packed_data = FormageChange(self.__vector_len__).run(
            data,
            src_format=src_format,
            dst_format=dst_format,
            tiling_info=self.generate_tiling(self.vars[name]))
        self.sess.run(tf.compat.v1.assign(self.vars[name], packed_data))

    def generate_tiling(self, value: tf.Variable) -> List[MatrixTiling]:
        return [MatrixTiling(0, 352), MatrixTiling(1, 128)]

    def check_require_folding(self, node, key: str):
        for attr in node.attrs:
            if attr.key == key:
                return False
        return True

    def get_variables(self) -> Dict[str, tf.Variable]:
        all_vars = tf.compat.v1.global_variables()
        return {v.name.split(':')[0]: v for v in all_vars}
