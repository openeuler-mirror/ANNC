from ._mlir_libs._kpgemm import MemType as MT
from ._mlir_libs._kpgemm import DataType as DT

class MemType:
    Undefined = MT.Undefined
    DRAM = MT.DRAM
    Global = MT.Global
    Constant_Cache = MT.Constant_Cache
    Texture = MT.Texture
    L2_Cache = MT.L2_Cache
    L1_Cache = MT.L1_Cache
    Local = MT.Local
    Shared = MT.Shared
    Register = MT.Register
    VectorReg = MT.VectorReg
    ScalarReg = MT.ScalarReg

class DataType:
    Unknown = DT.Unknown
    FP32 = DT.FP32
    FP16 = DT.FP16
    FP8 = DT.FP8
    FP64 = DT.FP64
    Int8 = DT.Int8
    Int16 = DT.Int16
    Int32 = DT.Int32
    Int64 = DT.Int64
    UInt8 = DT.UInt8
    UInt16 = DT.UInt16
    UInt32 = DT.UInt32
    UInt64 = DT.UInt64
    Bool = DT.Bool
    String = DT.String
    Complex64 = DT.Complex64
    Complex128 = DT.Complex128
