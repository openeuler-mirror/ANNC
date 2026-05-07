#include "Helper.h"

#include "mlir/Bytecode/BytecodeWriter.h"

namespace annc {
void outputCode(const ModuleOp &module, const std::string &filenameWithExt,
                int64_t elide) {
  if (module == nullptr) {
    llvm::report_fatal_error(llvm::StringRef("invalid module write to file"));
  }

  OpPrintingFlags flags;
  flags.enableDebugInfo();
  flags.elideLargeElementsAttrs(elide);
  std::string errorMessage;
  auto output = openOutputFile(filenameWithExt, &errorMessage);

  if (!output) {
    llvm::report_fatal_error(llvm::StringRef(errorMessage));
  }

  module->print(output->os(), flags);
  output->keep();
}

void outputBinary(const ModuleOp &module, const std::string &output) {
  std::error_code ec;
  llvm::raw_fd_ostream dest(output, ec, llvm::sys::fs::OF_None);

  if (ec) llvm::report_fatal_error("Could not open output file");

  mlir::FallbackAsmResourceMap fallbackResourceMap;
  mlir::ParserConfig config(module->getContext(), true, &fallbackResourceMap);
  mlir::BytecodeWriterConfig writerConfig(fallbackResourceMap);

  mlir::writeBytecodeToFile(module, dest, writerConfig);
  dest.flush();
}

llvm::ArrayRef<int64_t> getTensorShape(const Value &val) {
  auto type = dyn_cast<TensorType>(val.getType());
  if (type != nullptr) return type.getShape();
  auto mType = dyn_cast<MemRefType>(val.getType());
  if (mType != nullptr) return mType.getShape();
  return {};
}

NameLoc getLoc(MLIRContext *context, const std::string &name) {
  return NameLoc::get(StringAttr::get(context, name));
}
std::string getLocName(Operation *op) {
  auto loc = dyn_cast<::mlir::NameLoc>(op->getLoc());
  if (loc == nullptr) return "";
  return loc.getName().str();
}
std::string getValLocName(Value val) {
  auto loc = dyn_cast<::mlir::NameLoc>(val.getLoc());
  if (loc == nullptr) return "";
  return loc.getName().str();
}

std::string ArrayToStr(llvm::ArrayRef<int64_t> &vec) {
  std::string str = "[";
  for (size_t i = 0; i < vec.size(); i++) {
    str += std::to_string(vec[i]);
    if (i < vec.size() - 1) str += ",";
  }
  str += "]";
  return str;
}

std::string getStrValSafely(const Attribute &attr) {
  auto attr_ = dyn_cast<mlir::StringAttr>(attr);
  if (!attr_) return "";
  return attr_.str();
}
std::vector<std::string> StrArrayToSmallVector(const ArrayAttr &attr) {
  if (attr == nullptr) return {};
  std::vector<std::string> vec(attr.size());
  for (size_t i = 0; i < attr.size(); i++) {
    vec[i] = getStrValSafely(attr[i]);
  }
  return vec;
}

std::vector<std::string> splitString(const std::string &str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

int64_t getNumElements(Value val) {
  auto shape = getTensorShape(val);
  if (shape.size() == 0) return 0;
  return std::accumulate(shape.begin(), shape.end(), 1,
                         std::multiplies<int64_t>());
}

int64_t getElementSize(Value val) {
  auto type = dyn_cast<TensorType>(val.getType());
  if (type == nullptr) return {};
  Type eltType = type.getElementType();
  if (Float32Type fp32 = dyn_cast<Float32Type>(eltType))
    return sizeof(float);
  else if (Float16Type fp16 = dyn_cast<Float16Type>(eltType))
    return sizeof(float) / 2;
  else if (IntegerType it = dyn_cast<IntegerType>(eltType))
    return it.getWidth() / 8;
  return 0;
}

int64_t getSize(Value val) { return getNumElements(val) * getElementSize(val); }

std::string getBaseName(std::string path) {
  auto pos = path.find_last_of("/");
  if (pos != std::string::npos) path = path.substr(pos + 1);
  pos = path.find_last_of(".");
  if (pos != std::string::npos) path = path.substr(0, pos);
  return path;
}
std::string getOpTyName(Operation *op) {
  std::string name = op->getName().getStringRef().str();
  auto pos = name.find_last_of(".");
  return name.substr(pos + 1);
}

void updateFuncOp(func::FuncOp funcOp) {
  if (funcOp == nullptr) return;
  func::ReturnOp retOp =
      *funcOp.getBody().back().getOps<func::ReturnOp>().begin();
  llvm::SmallVector<Type> outputTypes{};
  std::for_each(retOp.getOperands().begin(), retOp.getOperands().end(),
                [&](Value val) {
                  TensorType type = cast<TensorType>(val.getType());
                  outputTypes.push_back(type);
                });
  llvm::SmallVector<Type> inputTypes{};
  auto inputVals = funcOp.getBody().front().getArguments();
  std::for_each(inputVals.begin(), inputVals.end(), [&](Value val) {
    TensorType type = cast<TensorType>(val.getType());
    inputTypes.push_back(type);
  });
  OpBuilder builder(funcOp.getContext());
  funcOp.setType(builder.getFunctionType(inputTypes, outputTypes));
}

func::ReturnOp getReturnOpFromFunc(func::FuncOp funcOp) {
  func::ReturnOp ret = nullptr;
  funcOp.walk([&](func::ReturnOp retOp) { ret = retOp; });
  return ret;
}
}  // namespace annc