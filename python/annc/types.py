from . import ir as mlir
from .enums import MemType, DataType
from ._mlir_libs._annc import TensorType as TT
from ._mlir_libs._annc import get_value_type as get_vt
from ._mlir_libs._annc import get_tensor_type as get_tt
from ._mlir_libs._annc import get_fp32_tensor_type as get_fp32_t
from ._mlir_libs._annc import get_fp64_tensor_type as get_fp64_t
from ._mlir_libs._annc import get_int_tensor_type as get_int_t
from ._mlir_libs._annc import get_int64_tensor_type as get_int64_t
from ._mlir_libs._annc import wrap_tensor_type as wrap_tt
from ._mlir_libs._annc import clone_tensor_with_name as clone_twn
from ._mlir_libs._annc import wrap_tiling_attr as wrap_ta
from ._mlir_libs._annc import TensorPtr as TP
from ._mlir_libs._annc import get_tensor_ptr as get_tp
from ._mlir_libs._annc import TilingAttr as TA
from ._mlir_libs._annc import get_tiling_attr as get_ta
from ._mlir_libs._annc import get_mem_type_attr
from typing import List, Union
import numpy as np


class ElemType:
    FP16 = mlir.F16Type.get
    FP32 = mlir.F32Type.get
    FP64 = mlir.F64Type.get
    SInt = mlir.IntegerType.get_signed
    Int = mlir.IntegerType.get_signless

NUMPY_DTYPE = {
    np.int32: ElemType.SInt,
    np.float32: ElemType.FP32,
    np.float64: ElemType.FP64,
}


class TilingAttr(TA):
    def get(axes: List[int], start: List[List[int]], size: List[List[int]]) -> 'TilingAttr':
        #  64  C++ 
        axes_ = mlir.ArrayAttr.get([
            mlir.IntegerAttr.get(mlir.IntegerType.get_signless(64), i) for i in axes])
        start_ = mlir.ArrayAttr.get([mlir.ArrayAttr.get([
            mlir.IntegerAttr.get(mlir.IntegerType.get_signless(64), i)
            for i in si]) for si in start])
        size_ = mlir.ArrayAttr.get([mlir.ArrayAttr.get([
            mlir.IntegerAttr.get(mlir.IntegerType.get_signless(64), i)
            for i in si]) for si in size])

        return get_ta(axes_, start_, size_)
    def get_axes(self) -> mlir.ArrayAttr: ...
    def get_start(self) -> mlir.ArrayAttr: ...
    def get_size(self) -> mlir.ArrayAttr: ...


def intra_tiling(axes: List[int], start: List[List[int]], size: List[List[int]]):
    return {'onchip_tiling': TilingAttr.get(axes, start, size)}


def inter_tiling(axes: List[int], start: List[List[int]], size: List[List[int]]) -> TilingAttr:
    return {'device_tiling': TilingAttr.get(axes, start, size)}


class TensorType(TT):
    def get(shape: List[int], elem_type: mlir.Type, name: str = None,
            encoding: List[str] = None, stride: List[int] = None,
            layout: str = None, mem_type: MemType = None,
            address: int = None, device_tiling: TilingAttr = None,
            onchip_tiling: TilingAttr = None) -> 'TensorType':
        name = mlir.StringAttr.get(name) \
            if name is not None else mlir.BoolAttr.get(False)
        encoding = mlir.ArrayAttr.get([mlir.StringAttr.get(i) for i in encoding]) \
            if encoding is not None else mlir.BoolAttr.get(False)
        stride = mlir.ArrayAttr.get([mlir.IntegerAttr.get(mlir.IntegerType.get_signed(64), i) for i in stride]) \
            if stride is not None else mlir.BoolAttr.get(False)
        layout = mlir.StringAttr.get(layout) \
            if layout is not None else mlir.BoolAttr.get(False)
        mem_type = get_mem_type_attr(elem_type.context, mem_type) \
            if mem_type is not None else mlir.BoolAttr.get(False)
        address = mlir.IntegerAttr.get(mlir.IntegerType.get_signed(64), address) \
            if address is not None else mlir.StringAttr.get("")
        device_tiling = device_tiling or mlir.BoolAttr.get(False)
        onchip_tiling = onchip_tiling or mlir.BoolAttr.get(False)
        return get_tt(shape, elem_type, name, encoding, stride, layout, mem_type, address, device_tiling, onchip_tiling)
    def get_data(data: np.ndarray) -> 'TensorType':
        if data.dtype.type == np.float32:
            return get_fp32_t(list(data.shape), ElemType.FP32(), data.flatten().tolist())
        if data.dtype.type == np.float64:
            return get_fp64_t(list(data.shape), ElemType.FP64(), data.flatten().tolist())
        if data.dtype.type == np.int32:
            return get_int_t(list(data.shape), ElemType.SInt(), data.flatten().tolist())
        raise TypeError(f'Invalid data type: {data.dtype.type}')
    def get_shape(self) -> List[int]: ...
    def get_element_type(self) -> str: ...
    def get_name(self) -> str: ...
    def get_address(self) -> int: ...
    def get_layout(self) -> str: ...
    def get_stride(self) -> List[int]: ...
    def get_mem_type(self) -> MemType: ...
    def get_device_parallel(self) -> TilingAttr: ...
    def get_onchip_parallel(self) -> TilingAttr: ...
    def clone(self) -> 'TensorType': ...
    def clone_raw_type(self) -> 'TensorType': ...
    def clone_with_name(self, name: mlir.StringAttr) -> 'TensorType': ...
    def clone_with_device_tiling(self) -> 'TensorType': ...
    def clone_with_onchip_tiling(self) -> 'TensorType': ...

def get_tensor_type(value: mlir.Value) -> TensorType:
    return get_vt(value)

def wrap_type(tt) -> mlir.Type:
    return wrap_tt(tt)

def clone_tensor_with_name(tt: TensorType, name: str) -> TensorType:
    return clone_twn(tt, name)

def wrap_tiling_attr(attr) -> mlir.Attribute:
    return wrap_ta(attr)

from ._mlir_libs._annc import set_tensor_name
from ._mlir_libs._annc import set_tensor_encoding
from ._mlir_libs._annc import set_tensor_stride
from ._mlir_libs._annc import set_tensor_layout
from ._mlir_libs._annc import set_tensor_mem_type
from ._mlir_libs._annc import set_tensor_address
from ._mlir_libs._annc import set_tensor_device_parallel
from ._mlir_libs._annc import set_tensor_onchip_parallel

from ._mlir_libs._annc import set_value_type

from ._mlir_libs._annc import replace_all_uses_with as repl_uses_with
from ._mlir_libs._annc import replace_all_uses_except as repl_uses_except


class Value:
    def __init__(self, value: mlir.Value):
        self.value = value
        self.tensor_type = get_tensor_type(value)
    
    def set_name(self, name: str):
        set_tensor_name(self.tensor_type, mlir.StringAttr.get(name))
    
    def set_encoding(self, encoding: List[str]):
        set_tensor_encoding(self.tensor_type, mlir.ArrayAttr.get([
            mlir.StringAttr.get(i) for i in encoding]))
    
    def set_stride(self, stride: List[int]):
        set_tensor_stride(self.tensor_type, mlir.ArrayAttr.get([
            mlir.IntegerAttr.get(mlir.IntegerType.get_signed(64), i)
            for i in stride]))
    
    def set_layout(self, layout: str):
        set_tensor_layout(self.tensor_type, mlir.StringAttr.get(layout))
    
    def set_mem_type(self, mem_type: MemType):
        set_tensor_mem_type(self.tensor_type, mem_type)

    def set_address(self, address: int):
        set_tensor_address(self.tensor_type, 
            mlir.IntegerAttr.get(mlir.IntegerType.get_signed(64), address))

    def set_device_tiling(self, tiling):
        if not isinstance(tiling, mlir.Attribute):
            try:
                tiling = wrap_tiling_attr(tiling)
            except (SystemError, RuntimeError):
                return
        set_tensor_device_parallel(self.tensor_type, tiling)

    def set_device_parallel(self, axes: List[int], starts: List[List[int]], sizes: List[List[int]]):
        tiling = TilingAttr.get(axes, starts, sizes)
        self.set_device_tiling(tiling)
    
    def set_onchip_tiling(self, tiling):
        if not isinstance(tiling, mlir.Attribute):
            try:
                tiling = wrap_tiling_attr(tiling)
            except (SystemError, RuntimeError):
                return
        set_tensor_onchip_parallel(self.tensor_type, tiling)
        
    def set_onchip_parallel(self, axes: List[int], starts: List[List[int]], sizes: List[List[int]]):
        tiling = TilingAttr.get(axes, starts, sizes)
        self.set_onchip_tiling(tiling)

    def set_tensor_type(self, type: TensorType):
        set_value_type(self.value, type)

def get_value_ptr(value: mlir.Value) -> Value:
    return Value(value)


class Operation:
    def __init__(self, op: mlir.Operation):
        self.op = op
    
    @property
    def name(self):
        return self.op.name
    
    def get_operand(self, i: int) -> Value:
        return Value(self.op.operands[i])
    
    def get_result(self, i: int) -> Value:
        return Value(self.op.results[i])


def get_op_ptr(op: Union[mlir.Operation, mlir.OpView]):
    if isinstance(op, mlir.Operation):
        return Operation(op)
    return Operation(op.operation)


class TensorPtr(TP):
    name: str = TP.name
    shape: List[int] = TP.shape
    dtype: DataType = TP.dtype
    address: int = TP.address
    layout: str = TP.layout
    stride: List[int] = TP.stride


def get_tensor_ptr(value: mlir.Value) -> TensorPtr:
    return get_tp(value)

def replace_all_uses_with(origin_val: mlir.Value, with_val: mlir.Value):
    repl_uses_with(origin_val, with_val)

def replace_all_uses_except(origin_val: mlir.Value, with_val: mlir.Value, except_op: mlir.Operation):
    repl_uses_except(origin_val, with_val, except_op)

def replace_operands(src: mlir.Operation, dst: mlir.Operation, in_map: dict, out_map: dict):
    for i, operand in enumerate(dst.operands):
        if operand in in_map:
            dst.operands[i] = in_map[operand]
    for src_out, dst_out in zip(src.results, dst.results):
        in_map[src_out] = dst_out
        if src_out in out_map:
            call_out = out_map[src_out]
            replace_all_uses_with(call_out, dst_out)
        
    for src_region, dst_region in zip(src.regions, dst.regions):
        for src_block, dst_block in zip(src_region.blocks, dst_region.blocks):
            for src_op, dst_op in zip(src_block.operations, dst_block.operations):
                replace_operands(src_op.operation, dst_op.operation, in_map, out_map)

def func_inline(call_op: mlir.Operation, func_op: mlir.Operation):
    args = func_op.regions[0].blocks[0].arguments
    ops = func_op.regions[0].blocks[0].operations
    ret_operands = ops[len(ops) - 1].operands
    
    input_map, output_map = {}, {}
    for call_in, func_in in zip(call_op.operands, args):
        input_map[func_in] = call_in
    for call_ret, func_ret in zip(call_op.results, ret_operands):
        output_map[func_ret] = call_ret
        
    with mlir.InsertionPoint(call_op):
        for i, op in enumerate(ops):
            if i == len(ops) - 1:
                break
            new_op = op.operation.clone()
            replace_operands(op.operation, new_op.operation, input_map, output_map)
    
    call_op.erase()
    func_op.erase()
