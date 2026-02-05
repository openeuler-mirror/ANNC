from . import ir as mlir
from .dialects import atir, func
from .types import TensorType, wrap_type
from .types import Value, get_value_ptr
from .types import ElemType, NUMPY_DTYPE
from typing import List
import numpy as np


class state:
    ip: mlir.InsertionPoint = None


class ForLoop:
    loop_ops = []
    return_vals = []
    
    def __call__(self, lower: int, 
                 upper: int, 
                 step: int, 
                 init_args: List[Value] = [], 
                 axis: int = None, 
                 post_op: str = None, 
                 parallel: bool = None, 
                 return_vals: List[Value] = []):
        self.return_vals.append(return_vals)
        with mlir.Location.unknown():
            init_args = [i.value for i in init_args]
            if axis is not None:
                axis = mlir.IntegerAttr.get(mlir.IntegerType.get_signless(64), axis)
            if post_op is not None:
                post_op = mlir.StringAttr.get(post_op)
            if parallel is not None:
                parallel = mlir.BoolAttr.get(parallel)
            loop_op = atir.ForOp([], 0, [], start=lower, end=upper, step=step, axis=axis, 
                                postOp=post_op, parallel=parallel, ip=state.ip)
            block: mlir.Block = mlir.Block.create_at_start(
                loop_op.region, arg_types=[mlir.IndexType.get()])
            state.ip = mlir.InsertionPoint(block)
            self.loop_ops.append(loop_op)
        return self
 
    def __enter__(self):
        return self
    
    @property
    def index(self) -> Value:
        return get_value_ptr(self.loop_ops[-1].region.blocks[0].arguments[0])

    @property
    def loop(self) -> atir.ForOp:
        return self.loop_ops[-1]
    
    def clone_new_loop(self, ret_types: List[mlir.Type], ip: mlir.InsertionPoint = None) -> atir.ForOp:
        with mlir.Location.unknown():
            loop = atir.ForOp(ret_types, 0, [], start=self.loop.start, end=self.loop.end, 
                             step=self.loop.step, axis=self.loop.axis, postOp=self.loop.postOp, 
                             parallel=self.loop.parallel, ip=ip)
            # loop.operation.move_after(self.loop.operation)
            block: mlir.Block = mlir.Block.create_at_start(loop.region, arg_types=[mlir.IndexType.get()])
            origin_block: mlir.Block = self.loop.regions[0].blocks[0]
            origin_arg = origin_block.arguments[0]
            values = {origin_arg: block.arguments[0]}
            with mlir.InsertionPoint(block):
                has_return = False
                for op in origin_block.operations:
                    if isinstance(op, atir.ReturnOp):
                        has_return = True
                    new_op = op.operation.clone()
                    for i, operand in enumerate(new_op.operands):
                        if operand in values:
                            new_op.operands[i] = values[operand]
                    for i, res in enumerate(new_op.results):
                        values[op.results[i]] = res
                
                if not has_return:
                    atir.ReturnOp([])
    
        return loop
    
    def __exit__(self, exc_type, exc_value, traceback):
        ret_types: List[mlir.Type] = []
        for op in state.ip.block.operations:
            if isinstance(op, atir.ReturnOp):
                for operand in op.operands:
                    ret_types.append(operand.type)
        
        state.ip = mlir.InsertionPoint(self.loop.operation)
        loop = self.clone_new_loop(ret_types, state.ip)
        loop.operation.move_after(self.loop.operation)
        self.loop.operation.attributes['remove'] = mlir.BoolAttr.get(True)
        
        if (len(loop.results) > 0):
            self.return_vals[-1].extend([get_value_ptr(i) for i in loop.results])
        
        # self.loop.operation.erase()
        has_return = False
        for op in state.ip.block.operations:
            if isinstance(op, func.ReturnOp):
                state.ip = mlir.InsertionPoint(op.operation)
                has_return = True
        if not has_return:
            state.ip = mlir.InsertionPoint(state.ip.block)
        
        self.loop_ops.pop()
        self.return_vals.pop()


class Atir:
    def __init__(self):
        self.Loc = mlir.Location
        self.for_loop = ForLoop()

    def none(self) -> Value:
        return get_value_ptr(atir.NoneOp(
            mlir.NoneType.get(), loc=self.Loc.unknown(), ip=state.ip).results[0])

    def matmul(self, output: TensorType, lhs: Value, rhs: Value, bias: Value = None) -> Value:
        bias_value = bias.value if bias is not None else self.none().value
        return get_value_ptr(atir.MatMulOp(
            wrap_type(output), lhs, rhs, bias_value, loc=self.Loc.unknown(), ip=state.ip).output)

    def add(self, output: TensorType, lhs: Value, rhs: Value, scalar: int = None) -> Value:
        return get_value_ptr(atir.AddOp(wrap_type(output), lhs.value, rhs.value, scalar, 
                                       loc=self.Loc.unknown(), ip=state.ip).output)

    def relu(self, output:TensorType, input: Value):
        return get_value_ptr(atir.ReluOp(wrap_type(output), input.value,
                             loc=self.Loc.unknown(), ip=state.ip).output)

    def ret(self, ret_vals: List[Value]):
        atir.ReturnOp([val.value for val in ret_vals], loc=self.Loc.unknown(), ip=state.ip)


def none(loc=None, **kwargs):
    loc = loc or mlir.Location.unknown()
    return Value(atir.NoneOp(mlir.NoneType.get(), loc=loc, ip=state.ip).output)


def ret(values: List[Value], loc=None) -> List[Value]:
    loc = loc or mlir.Location.unknown()
    op = atir.ReturnOp([i.value for i in values], loc=loc, ip=state.ip)
    return [Value(i) for i in op.results]


def constant(name: str = None, dtype: ElemType = None, shape: List[int] = None,
             data: np.ndarray = None, sym_visibility ='private', loc=None) -> Value:
    loc = loc or mlir.Location.unknown()
    if dtype is not None and shape is not None:
        return Value(atir.ConstantOp(wrap_type(TensorType.get(shape, dtype())),
                                     name, sym_visibility=sym_visibility,
                                     loc=loc, ip=state.ip).data)
    if data is not None:
        return Value(atir.ConstantOp(wrap_type(TensorType.get_data(data)),
                                     name, sym_visibility=sym_visibility,
                                     loc=loc, ip=state.ip).data)
    raise ValueError(f'Invalid input arguments for constant: {name}')


def matmul(dtype: ElemType, shape: List[int], lhs: Value, rhs: Value, C: Value , withBias: bool, loc=None, bias_add: Value = None, **kwargs) -> Value:
    loc = loc or mlir.Location.unknown()
    # Create a none value for bias when it's None
    bias_add_m = bias_add.value if bias_add is not None else bias_add
    return Value(atir.MatMulOp(wrap_type(TensorType.get(shape, dtype, **kwargs)),
                               lhs.value, rhs.value, C.value, withBias, bias = bias_add_m, loc=loc, ip=state.ip).output)

def add(dtype: ElemType, shape: List[int], operands: List[Value], do_relu: bool = False, relu_limit: float = -1.0, scalar: int = None, loc=None, **kwargs) -> Value:
    loc = loc or mlir.Location.unknown()

    return Value(atir.AddOp(wrap_type(TensorType.get(shape, dtype, **kwargs)),
                            [v.value for v in operands], do_relu=do_relu, relu_limit=relu_limit, scalar=scalar, loc=loc, ip=state.ip).output)

def relu(dtype: ElemType, shape: List[int], input: Value, relu_limit: float = 0.0, loc=None, **kwargs) -> Value:
    loc = loc or mlir.Location.unknown()
    return Value(atir.ReluOp(wrap_type(TensorType.get(shape, dtype, **kwargs)), 
                             input.value, relu_limit = relu_limit, loc=loc, ip=state.ip).output)
