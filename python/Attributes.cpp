#include "AtirModule.h"
#include "mlir/CAPI/Registration.h"

namespace py = pybind11;
namespace atir {
MlirAttribute getTilingAttr_(MlirAttribute axes, MlirAttribute starts, MlirAttribute sizes) {
  auto ctx = unwrap(axes).getContext();
  return wrap(TilingAttr::get(ctx, dyn_cast<ArrayAttr>(unwrap(axes)),
                              dyn_cast<ArrayAttr>(unwrap(starts)),
                              dyn_cast<ArrayAttr>(unwrap(sizes))));
}

MlirAttribute getMemTypeAttr_(MlirContext ctx, int64_t memType) {
  return wrap(MemTypeAttr::get(unwrap(ctx), MemType(memType)));
}

template <typename T>
MlirAttribute wrapAttribute_(T attr) {
  return wrap(attr);
}

void populateAttributeModule(py::module_ m) {
  m.doc() = "Atir Attributes";

  py::enum_<DataType>(m, "DataType")
    .value("Unknown", DataType::Unknown)
    .value("FP32", DataType::FP32)
    .value("FP16", DataType::FP16)
    .value("FP8", DataType::FP8)
    .value("FP64", DataType::FP64)
    .value("Int8", DataType::Int8)
    .value("Int16", DataType::Int16)
    .value("Int32", DataType::Int32)
    .value("Int64", DataType::Int64)
    .value("UInt8", DataType::UInt8)
    .value("UInt16", DataType::UInt16)
    .value("UInt32", DataType::UInt32)
    .value("UInt64", DataType::UInt64)
    .value("Bool", DataType::Bool)
    .value("String", DataType::String)
    .value("Complex64", DataType::Complex64)
    .value("Complex128", DataType::Complex128)
    .export_values();
  
  py::enum_<MemType>(m, "MemType")
    .value("Undefined", MemType::Undefined)
    .value("DRAM", MemType::DRAM)
    .value("Global", MemType::Global)
    .value("Constant_Cache", MemType::Constant_Cache)
    .value("Texture", MemType::Texture)
    .value("L2_Cache", MemType::L2_Cache)
    .value("L1_Cache", MemType::L1_Cache)
    .value("Local", MemType::Local)
    .value("Shared", MemType::Shared)
    .value("Register", MemType::Register)
    .value("VectorReg", MemType::VectorReg)
    .value("ScalarReg", MemType::ScalarReg)
    .export_values();

  py::class_<TilingAttr>(m, "TilingAttr")
    .def(py::init())
    .def("get_axes", &TilingAttr::getTilingAxes)
    .def("get_start", &TilingAttr::getTilingStart)
    .def("get_size", &TilingAttr::getTilingSize);
  
  m.def("get_tiling_attr", &getTilingAttr_, py::arg("axes"), py::arg("starts"), py::arg("sizes"));
  m.def("get_mem_type_attr", &getMemTypeAttr_, py::arg("ctx"), py::arg("mem_type"));

  m.def("wrap_tiling_attr", &wrapAttribute_<TilingAttr>, py::arg("tiling"));
  m.def("k_dynamic_", []() { return ShapedType::kDynamic; });
}

} // namespace atir