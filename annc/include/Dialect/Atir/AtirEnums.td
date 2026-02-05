//===----------------------------------------------------------------------===//
// Atir Enum definitions
//===----------------------------------------------------------------------===//
#ifndef ATIR_ENUMS_TD
#define ATIR_ENUMS_TD
include "mlir/IR/EnumAttr.td"

def Atir_MemType : I32EnumAttr<"MemType",
    "memory hierarchy descriptor",
    [
        I32EnumAttrCase<"Undefined", 99>,
        I32EnumAttrCase<"DRAM", 0>,
        I32EnumAttrCase<"Global", 1>,
        I32EnumAttrCase<"Constant_Cache", 2>,
        I32EnumAttrCase<"Texture", 3>,
        I32EnumAttrCase<"L2_Cache", 4>,
        I32EnumAttrCase<"L1_Cache", 5>,
        I32EnumAttrCase<"Local", 6>,
        I32EnumAttrCase<"Shared", 7>,
        I32EnumAttrCase<"Register", 8>,
        I32EnumAttrCase<"VectorReg", 9>,
        I32EnumAttrCase<"ScalarReg", 10>
    ]>{
    let genSpecializedAttr = 0;
    let cppNamespace = "::atir";
}

def Atir_DataType : I32EnumAttr<"DataType",
  "data type of the tensor",[
    I32EnumAttrCase<"Unknown", 99>,
    I32EnumAttrCase<"FP32", 0>,
    I32EnumAttrCase<"FP16", 1>,
    I32EnumAttrCase<"FP8", 2>,
    I32EnumAttrCase<"FP64", 3>,
    I32EnumAttrCase<"Int8", 4>,
    I32EnumAttrCase<"Int16", 5>,
    I32EnumAttrCase<"Int32", 6>,
    I32EnumAttrCase<"Int64", 7>,
    I32EnumAttrCase<"UInt8", 8>,
    I32EnumAttrCase<"UInt16", 9>,
    I32EnumAttrCase<"UInt32", 10>,
    I32EnumAttrCase<"UInt64", 11>,
    I32EnumAttrCase<"Bool", 12>,
    I32EnumAttrCase<"String", 13>,
    I32EnumAttrCase<"Complex64", 14>,
    I32EnumAttrCase<"Complex128", 15>
  ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::atir";
}

def Atir_Axis : I32EnumAttr<"Axis",
  "Axis of tensor shape",[
    I32EnumAttrCase<"Unknown", 99>,
    I32EnumAttrCase<"Batch", 0>,
    I32EnumAttrCase<"H", 1>,
    I32EnumAttrCase<"W", 2>,
    I32EnumAttrCase<"IC", 3>,
    I32EnumAttrCase<"OC", 4>,
    I32EnumAttrCase<"N", 5>,
    I32EnumAttrCase<"M", 6>,
    I32EnumAttrCase<"K", 7>
  ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::atir";
}

def Atir_TilingStragegy : I32EnumAttr<"TilingStragegy",
  "Tiling stragegy",[
    I32EnumAttrCase<"Unknown", 99>,
    I32EnumAttrCase<"Continuous", 0>,
    I32EnumAttrCase<"Stepwise", 1>
  ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::atir";
}

def Atir_RoundMode : I32EnumAttr<"RoundMode",
    "rounding mode of the operation",[
      I32EnumAttrCase<"RoundHalfUp", 0>,
      I32EnumAttrCase<"RoundFloor", 1>,
      I32EnumAttrCase<"RoundCeil", 2>,
      I32EnumAttrCase<"RoundTrunc", 3>,
      I32EnumAttrCase<"RoundEven", 4>
    ]>{
    let genSpecializedAttr = 1;
    let cppNamespace = "::atir";
}
#endif // ATIR_ENUMS_TD