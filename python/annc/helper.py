from . import ir as mlir
from .builder import MLIRBuilder
from .ops import state
from .types import get_tensor_type
from .types import wrap_type, TensorType, Value
from typing import List, Union
import numpy as np

class AtirBuilder(MLIRBuilder):
    def __init__(self, ctx: mlir.Context = None):
        super().__init__(ctx)


def get_sym_name(op: mlir.Operation):
    if 'sym_name' not in op.attributes:
        return ''
    return mlir.StringAttr(op.attributes['sym_name']).value


def set_sym_name(op: mlir.Operation, name: str):
    op.attributes['sym_name'] = mlir.StringAttr.get(name)


def lookup_func(module: mlir.Module, name: str) -> mlir.Operation:
    for op in module.body.operations:
        if get_sym_name(op) == name:
            return op.operation
    return None


def create_tensor_type(input: Union[dict, np.ndarray]):
    if isinstance(input, dict):
        name = name=input.get('name')
        shape = input['shape']
        dtype = input['dtype']
        return wrap_type(TensorType.get(shape, dtype(), name))
    elif isinstance(input, np.ndarray):
        return wrap_type(TensorType.get_data(input))
    raise TypeError(f'Invalid input argument: {input}')


class kp:
    def jit(func):
        def wrapper(*args, **kwargs):
            builder = AtirBuilder(kwargs.get('ctx', None))
            ip = kwargs.get('ip')
            name = kwargs.get('name', 'test')
            #module and func
            if ip is None:
                builder.create_module(name)
                func_op = builder.create_func(func.__name__)
                inputs = kwargs.get('inputs', [])
                input_types = []
                for i, input in enumerate(inputs):
                    input_val = func_op.entry_block.add_argument(
                        create_tensor_type(input), mlir.Location.unknown())
                    input_types.append(input_val.type)
                    kwargs['inputs'][i] = Value(input_val)
                func_op.attributes["function_type"] = mlir.TypeAttr.get(
                    mlir.FunctionType.get(inputs=tuple(input_types), results=()))
                func_op.attributes["llvm.emit_c_interface"] = mlir.UnitAttr.get()
                state.ip = builder.ip
            else:
                state.ip = ip

            #func region
            results = func(*args, **kwargs)
            #process return
            if ip is None:
                outputs: List[str] = kwargs.get('outputs', [])
                output_vals, output_types = [], []
                func_created = lookup_func(builder.m, func.__name__)
                for op in func_created.regions[0].blocks[0].operations:
                    if op.name == "atir.None":
                        continue;
                    for res in op.results:
                        t_name = get_tensor_type(res).get_name()
                        # if t_name in outputs:
                        #     output_vals.append(res)
                        #     output_types.append(res.type)
                builder.create_return(output_vals)
                func_op.attributes["function_type"] = mlir.TypeAttr.get(
                    mlir.FunctionType.get(inputs=func_op.attributes["function_type"].value.inputs,
                                          results=tuple(output_types)))
                # subprocess.check_call(['annc-asm', builder.m, '--passes'])
                return builder.m
            return results
        return wrapper
