#include "PimpModule.h"

namespace py = pybind11;

namespace pimp {
void replaceAllUsesWith_(MlirValue val1, MlirValue val2) {
  // replace val1's users with val2
  unwrap(val1).replaceAllUsesWith(unwrap(val2));
}
void replaceAllUsesExcept_(MlirValue val1, MlirValue val2, MlirOperation op) {
  unwrap(val1).replaceAllUsesExcept(unwrap(val2), unwrap(op));
}

TensorType getValueType_(MlirValue value) {
  auto type = unwrap(value).getType();

  if (type == nullptr) {
      return nullptr;
  }
  return dyn_cast<TensorType>(type);
}

TensorType getTensorType_(std::vector<int64_t> shape, MlirType elemType,
                          MlirAttribute name = {}, MlirAttribute encoding = {},
                          MlirAttribute stride = {}, MlirAttribute layout = {},
                          MlirAttribute memType = {},
                          MlirAttribute address = {},
                          MlirAttribute devoceParallel = {},
                          MlirAttribute onchipParallel = {}) {
  StringAttr name_;
  if (isa<StringAttr>(unwrap(name))) name_ = dyn_cast<StringAttr>(unwrap(name));
  Attribute encoding_;
  if (isa<ArrayAttr>(unwrap(encoding))) encoding_ = unwrap(encoding);
  ArrayAttr stride_;
  if (isa<ArrayAttr>(unwrap(stride)))
    stride_ = dyn_cast<ArrayAttr>(unwrap(stride));
  StringAttr layout_;
  if (isa<StringAttr>(unwrap(layout)))
    layout_ = dyn_cast<StringAttr>(unwrap(layout));
  MemTypeAttr memType_;
  if (isa<MemTypeAttr>(unwrap(memType)))
    memType_ = dyn_cast<MemTypeAttr>(unwrap(memType));
  IntegerAttr address_;
  if (isa<IntegerAttr>(unwrap(address)))
    address_ = dyn_cast<IntegerAttr>(unwrap(address));
  TilingAttr devoceParallel_;
  if (isa<TilingAttr>(unwrap(devoceParallel)))
    devoceParallel_ = dyn_cast<TilingAttr>(unwrap(devoceParallel));
  TilingAttr onchipParallel_;
  if (isa<TilingAttr>(unwrap(onchipParallel)))
    onchipParallel_ = dyn_cast<TilingAttr>(unwrap(onchipParallel));
  return TensorType::get(shape, unwrap(elemType), name_, encoding_, stride_,
                         layout_, memType_, address_, devoceParallel_,
                         onchipParallel_);
}

template <typename T>
TensorType getDenseTensorType_(std::vector<int64_t> shape, MlirType elemType,
                               std::vector<T> data) {
  auto elems = DenseElementsAttr::get(
      RankedTensorType::get(shape, unwrap(elemType)), ArrayRef<T>(data));
  return TensorType::get(shape, unwrap(elemType), elems);
}
TensorType getFloatTensorType_(std::vector<int64_t> shape, MlirType elemType,
                               std::vector<float_t> data) {
  return getDenseTensorType_<float_t>(shape, elemType, data);
}
TensorType getFloat64TensorType_(std::vector<int64_t> shape, MlirType elemType,
                                 std::vector<double_t> data) {
  return getDenseTensorType_<double_t>(shape, elemType, data);
}
TensorType getIntTensorType_(std::vector<int64_t> shape, MlirType elemType,
                             std::vector<int32_t> data) {
  return getDenseTensorType_<int32_t>(shape, elemType, data);
}
TensorType getInt64TensorType_(std::vector<int64_t> shape, MlirType elemType,
                               std::vector<int64_t> data) {
  return getDenseTensorType_<int64_t>(shape, elemType, data);
}

void setTensorName_(TensorType type, MlirAttribute name) {
  type.setName(dyn_cast<StringAttr>(unwrap(name)));
}
void setTensorEncoding_(TensorType type, MlirAttribute encoding) {
  type.setEncoding(unwrap(encoding));
}
void setTensorStride_(TensorType type, MlirAttribute stride) {
  type.setStride(dyn_cast<ArrayAttr>(unwrap(stride)));
}
void setTensorLayout_(TensorType type, MlirAttribute layout) {
  type.setLayout(dyn_cast<StringAttr>(unwrap(layout)));
}
void setTensorMemType_(TensorType type, int64_t memType) {
  type.setMemType(MemTypeAttr::get(type.getContext(), MemType(memType)));
}
void setTensorAddress_(TensorType type, MlirAttribute address) {
  type.setAddress(dyn_cast<IntegerAttr>(unwrap(address)));
}
void setTensorDeviceParallel_(TensorType type, MlirAttribute tiling) {
  type.setDeviceParallel(dyn_cast<TilingAttr>(unwrap(tiling)));
}
void setTensorOnchipParallel_(TensorType type, MlirAttribute tiling) {
  type.setOnchipParallel(dyn_cast<TilingAttr>(unwrap(tiling)));
}

TensorType cloneWithName_(TensorType type, std::string name) {
  return type.cloneWithName(mlir::StringAttr::get(type.getContext(), name));
}

struct TensorPtr {
  std::string name;
  std::vector<int64_t> shape;
  DataType dtype{DataType::Unknown};
  int64_t address{-1};
  std::string layout;
  std::vector<int64_t> stride;
};

TensorPtr getTensorPtr_(MlirValue value) {
  TensorPtr tp;
  auto type = unwrap(value).getType();
  if (type == nullptr) return tp;
  auto tt = dyn_cast<TensorType>(type);
  tp.name = tt.getValueOfName();
  tp.shape = tt.getValueOfShape();
  tp.layout = tt.getValueOfLayout();
  tp.stride = tt.getValueOfStride();
  tp.address = tt.getValueOfAddress();
  if (tt.getElementType().isF32()) {
    tp.dtype = DataType::FP32;
  } else if (tt.getElementType().isF16()) {
    tp.dtype = DataType::FP16;
  } else if (tt.getElementType().isF64()) {
    tp.dtype = DataType::FP64;
  } else if (tt.getElementType().isSignedInteger(8)) {
    tp.dtype = DataType::Int8;
  } else if (tt.getElementType().isSignedInteger(16)) {
    tp.dtype = DataType::Int16;
  } else if (tt.getElementType().isSignedInteger(32)) {
    tp.dtype = DataType::Int32;
  }
  return tp;
}

void setValueType_(MlirValue val, TensorType ttype) {
  Value value = unwrap(val);
  value.setType(ttype);
}

template <typename T>
MlirType wrapType_(T type) {
  return wrap(type);
}

void populateTypesModule(py::module_ m) {
  py::class_<TensorType>(m, "TensorType")
      .def(py::init())
      .def("get_shape", &TensorType::getValueOfShape)
      .def("get_element_type", &TensorType::getValueOfElementType)
      .def("get_name", &TensorType::getValueOfName)
      .def("get_mem_type", &TensorType::getValueOfMemType)
      .def("get_address", &TensorType::getValueOfAddress)
      .def("get_layout", &TensorType::getValueOfLayout)
      .def("get_stride", &TensorType::getValueOfStride)
      .def("get_device_parallel", &TensorType::getDeviceParallel)
      .def("get_onchip_parallel", &TensorType::getOnchipParallel)
      .def("clone", &TensorType::clone)
      .def("clone_raw_type", &TensorType::cloneRawType)
      .def("clone_with_name", &TensorType::cloneWithName)
      .def("clone_with_device_tiling", &TensorType::cloneWithDeviceTiling)
      .def("clone_with_onchip_tiling", &TensorType::cloneWithOnchipTiling);

  m.def("get_value_type", &getValueType_, py::arg("value"));
  m.def("get_tensor_type", &getTensorType_);
  m.def("get_fp32_tensor_type", &getFloatTensorType_);
  m.def("get_fp64_tensor_type", &getFloat64TensorType_);
  m.def("get_int_tensor_type", &getIntTensorType_);
  m.def("get_int64_tensor_type", &getInt64TensorType_);
  m.def("set_value_type", &setValueType_, py::arg("value"),
        py::arg("tensor_type"));

  m.def("clone_tensor_with_name", &cloneWithName_, py::arg("tensor"),
        py::arg("name"));
  m.def("replace_all_uses_with", &replaceAllUsesWith_, py::arg("origin_val"),
        py::arg("new_val"));
  m.def("replace_all_uses_except", &replaceAllUsesExcept_,
        py::arg("origin_val"), py::arg("new_val"), py::arg("except_op"));

  m.def("set_tensor_name", &setTensorName_, py::arg("tensor"), py::arg("name"));
  m.def("set_tensor_encoding", &setTensorEncoding_, py::arg("tensor"),
        py::arg("encoding"));
  m.def("set_tensor_stride", &setTensorStride_, py::arg("tensor"),
        py::arg("stride"));
  m.def("set_tensor_layout", &setTensorLayout_, py::arg("tensor"),
        py::arg("layout"));
  m.def("set_tensor_mem_type", &setTensorMemType_, py::arg("tensor"),
        py::arg("mem_type"));
  m.def("set_tensor_address", &setTensorAddress_, py::arg("tensor"),
        py::arg("address"));
  m.def("set_tensor_device_parallel", &setTensorDeviceParallel_,
        py::arg("tensor"), py::arg("tiling"));
  m.def("set_tensor_onchip_parallel", &setTensorOnchipParallel_,
        py::arg("tensor"), py::arg("tiling"));

  py::class_<TensorPtr>(m, "TensorPtr")
      .def(py::init())
      .def_readwrite("name", &TensorPtr::name)
      .def_readwrite("shape", &TensorPtr::shape)
      .def_readwrite("dtype", &TensorPtr::dtype)
      .def_readwrite("address", &TensorPtr::address)
      .def_readwrite("layout", &TensorPtr::layout)
      .def_readwrite("stride", &TensorPtr::stride);

  m.def("get_tensor_ptr", &getTensorPtr_, py::arg("value"));

  m.def("wrap_tensor_type", &wrapType_<TensorType>, py::arg("tensor_type"));
}
}  // namespace pimp