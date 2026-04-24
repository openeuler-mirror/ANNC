from . import ir as mlir
from .dialects import func, pimp
from .types import TensorType
from io import TextIOWrapper
from typing import Union, List

class MLIRBuilder:
    def __init__(self, ctx: mlir.Context = None) -> None:
        self.ctx = mlir.Context() if ctx is None else ctx
        self.ctx.__enter__()
        pimp.register_dialect()
        self.ctx.load_all_available_dialects()
        
        self.loc = mlir.Location.unknown()
        
        self.m : mlir.Module = None
    
    def parse_module(self, f: Union[str, TextIOWrapper]) -> mlir.Module:
        if isinstance(f, TextIOWrapper):
            self.m = mlir.Module.parse(f.read())
        with open(f, 'r') as fp:
            self.m = mlir.Module.parse(fp.read())
        return self.m
    
    def create_module(self, name):
        self.m = mlir.Module.create(self.loc)
        self.m.operation.attributes['sym_name'] = mlir.StringAttr.get(name)
        return self
    
    def create_func(self, func_name: str, is_private: bool = True) -> func.FuncOp:
        visibility = 'private' if is_private else None
        fop = func.FuncOp(func_name, ((), ()), visibility=visibility, loc=self.loc)
        self.m.body.append(fop)
        mlir.Block.create_at_start(fop.regions[0])
        self.ip = mlir.InsertionPoint(fop.entry_block)
        return fop

    def create_main(self) -> func.FuncOp:
        return self.create_func("main", False)

    def create_return(self, outputs: List[TensorType]):
        func.ReturnOp(outputs, loc=self.loc, ip=self.ip)

    def __del__(self):
        self.ctx.__exit__(None, None, None)


if __name__ == '__main__':
    builder = MLIRBuilder()
    builder.create_module('demo')
    builder.create_main()
    
    # insert operations here
        
    builder.create_return([])
    with open('demo.mlir', 'w') as f:
        builder.m.operation.print(f, large_elements_limit=16)
