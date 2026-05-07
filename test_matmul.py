#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

from annc.ops import Atir, Value, TensorType
from annc.types import ElemType

def test_matmul_without_bias():
    """ bias  matmul """
    print(" matmul  bias...")
    
    #  Atir 
    atir = Atir()
    
    # 
    input_type = TensorType.get([2, 3], ElemType.FP32)
    weight_type = TensorType.get([3, 2], ElemType.FP32)
    output_type = TensorType.get([2, 2], ElemType.FP32)
    
    #  None 
    lhs = None  #  Value 
    rhs = None  #  Value 
    
    try:
        #  Atir.matmul 
        result = atir.matmul(output_type, lhs, rhs)
        print(" Atir.matmul ")
        print(f": {type(result)}")
        return True
    except Exception as e:
        print(f" Atir.matmul : {e}")
        return False

def test_global_matmul():
    """ matmul """
    print("\n matmul ...")
    
    try:
        from annc.ops import matmul, Value
        
        # 
        input_type = TensorType.get([2, 3], ElemType.FP32)
        weight_type = TensorType.get([3, 2], ElemType.FP32)
        output_type = TensorType.get([2, 2], ElemType.FP32)
        
        #  None 
        lhs = None  #  Value 
        rhs = None  #  Value 
        
        #  matmul 
        result = matmul(ElemType.FP32, [2, 2], lhs, rhs)
        print("  matmul ")
        print(f": {type(result)}")
        return True
    except Exception as e:
        print(f"  matmul : {e}")
        return False

def test_none_function():
    """ none """
    print("\n none ...")
    
    try:
        from annc.ops import none
        
        #  none 
        result = none()
        print(" none ")
        print(f": {type(result)}")
        print(f": {result.value}")
        return True
    except Exception as e:
        print(f" none : {e}")
        return False

if __name__ == "__main__":
    print(" Atir_MatMulOp  Python ...")
    
    test_none_function()
    test_matmul_without_bias()
    test_global_matmul()
    
    print("\n")