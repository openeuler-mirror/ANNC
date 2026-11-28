#include "Dialect/Atir/OpVerify/IoDef.h"
#include "Dialect/Atir/AtirOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"
#include <filesystem>
#include <vector>

#define DEFAULT_INPUT_DIM 4

using namespace llvm;
using namespace mlir;

namespace atir {

void IoTensorDef::addInput(const std::string &name, mlir::Value value) { 
  inputs.emplace_back(name, value);
}

const IoTensorDef::TensorStorage &IoTensorDef::getInputs() const { return inputs; }

const IoTensorDef::TensorStorage &IoTensorDef::getOutputs() const { return outputs;}

void IoTensorDef::addOutput(const std::string &name, mlir::Value value) {
  outputs.emplace_back(name, value);
}

bool IoTensorDef::isEmpty() const { return inputs.empty() && outputs.empty(); }

size_t IoTensorDef::getInputCount() const { return inputs.size(); }

size_t IoTensorDef::getOutputCount() const { return outputs.size(); }

std::string getFuncNameFromPath(const std::string &pathValue) {
  namespace fs = std::filesystem;
  fs::path p(pathValue);
  if (pathValue.empty() || !fs::exists(p)) {
    return "";
  }
  std::string baseName = p.stem().string();
  return baseName;
}

void createIoDef(mlir::ModuleOp root, IoTensorDef *inputs, IoTensorDef *outputs,
                 const std::string &funcName) {
  if (!root || !inputs) {
    llvm::errs() << "Error: Invalid parameters in createIoDef\n";
    return;
  }

  auto targetFuncOp = root.lookupSymbol<func::FuncOp>(funcName);
  if (!targetFuncOp) {
    llvm::errs() << "Error: Could not find target function: " << funcName << "\n";
    return;
  }

  llvm::dbgs() << "Creating input/output definitions for function: " << funcName << "\n";

  // Handle function arguments
  MutableArrayRef<BlockArgument> graphInputs;
  if (targetFuncOp.getNumArguments() > 0) {
    graphInputs = targetFuncOp.getArguments();
  }

  int inputIndex = 0;
  for (mlir::Value &arg : graphInputs) {
    std::string inputName = "input_" + std::to_string(inputIndex++);

    mlir::Type type = arg.getType();
    llvm::dbgs() << "Processing " << inputName << " argument type: " << type << "\n";

    auto tensorType = llvm::dyn_cast<atir::TensorType>(type);
    if (!tensorType) {
      llvm::dbgs() << "Info: tensorType is null! continue process other op...\n";
      continue;
    }

    mlir::ArrayRef<int64_t> shape = tensorType.getShape();
    mlir::Type elementType = tensorType.getElementType();

    int64_t numElements = 1;
    std::vector<int64_t> staticShape;
    for (int64_t dim : shape) {
      if (dim == mlir::ShapedType::kDynamic) {
        llvm::dbgs() << "Info: dynamic shape argument, using DEFAULT_INPUT_DIM. \n";
        dim = DEFAULT_INPUT_DIM;
      }
      numElements *= dim;
      staticShape.push_back(dim);
    }

    if (staticShape.empty()) continue;

    mlir::RankedTensorType standardTensorType =
        mlir::RankedTensorType::get(staticShape, elementType);

    mlir::Attribute randomAttr;

    if (mlir::isa<mlir::FloatType>(elementType)) {
      // Handle float types
      if (elementType.isF32()) {
        std::vector<float> randomData(numElements);
        for (int64_t i = 0; i < numElements; ++i) {
          randomData[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0f;
        }
        randomAttr = mlir::DenseElementsAttr::get(standardTensorType, 
                                                   llvm::ArrayRef<float>(randomData.data(), randomData.size()));
      } else if (elementType.isF64()) {
        std::vector<double> randomData(numElements);
        for (int64_t i = 0; i < numElements; ++i) {
          randomData[i] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX) * 10.0;
        }
        randomAttr = mlir::DenseElementsAttr::get(standardTensorType, 
                                                   llvm::ArrayRef<double>(randomData.data(), randomData.size()));
      } else if (elementType.isF16()) {
        // Handle F16 type
        std::vector<float> randomData(numElements);
        for (int64_t i = 0; i < numElements; ++i) {
          randomData[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0f;
        }
        randomAttr = mlir::DenseElementsAttr::get(standardTensorType, 
                                                   llvm::ArrayRef<float>(randomData.data(), randomData.size()));
      } else {
        llvm::errs() << "Warning: Unsupported float type: " << elementType << "\n";
      }
      
    } else if (mlir::isa<mlir::IntegerType>(elementType) || mlir::isa<mlir::IndexType>(elementType)) {
      unsigned bitWidth = 64;
      if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
        bitWidth = intType.getWidth();
      }
      if (bitWidth == 1) {  // Boolean type
        llvm::SmallVector<bool> randomData(numElements);
        for (int64_t i = 0; i < numElements; ++i) {
          randomData[i] = (rand() % 2) == 1;
        }
        randomAttr = mlir::DenseElementsAttr::get(standardTensorType, randomData);
      } else if (bitWidth == 8) {  // Int8 type
        std::vector<int8_t> randomData(numElements);
        for (int64_t i = 0; i < numElements; ++i) {
          randomData[i] = static_cast<int8_t>(rand() % 50);
        }
        randomAttr = mlir::DenseElementsAttr::get(standardTensorType, 
                                                   llvm::ArrayRef<int8_t>(randomData.data(), randomData.size()));
      } else if (bitWidth == 16) {  // Int16 type
        std::vector<int16_t> randomData(numElements);
        for (int64_t i = 0; i < numElements; ++i) {
          randomData[i] = static_cast<int16_t>(rand() % 100);
        }
        randomAttr = mlir::DenseElementsAttr::get(standardTensorType, 
                                                   llvm::ArrayRef<int16_t>(randomData.data(), randomData.size()));
      } else if (bitWidth == 32) {  // Int32 type
        std::vector<int32_t> randomData(numElements);
        for (int64_t i = 0; i < numElements; ++i) {
          randomData[i] = rand() % 100;
        }
        randomAttr = mlir::DenseElementsAttr::get(standardTensorType, 
                                                   llvm::ArrayRef<int32_t>(randomData.data(), randomData.size()));
      } else if (bitWidth == 64 || mlir::isa<mlir::IndexType>(elementType)) {  // Int64 type
        std::vector<int64_t> randomData(numElements);
        for (int64_t i = 0; i < numElements; ++i) {
          randomData[i] = rand() % 1000;
        }
        randomAttr = mlir::DenseElementsAttr::get(standardTensorType, 
                                                   llvm::ArrayRef<int64_t>(randomData.data(), randomData.size()));
      } else {
        llvm::errs() << "Warning: Unsupported integer bit width: " << bitWidth << "\n";
      }
      
    } else {
      llvm::errs() << "Warning: Unsupported element type for random data: " << elementType << "\n";
    }

    if (!randomAttr) {
      llvm::errs() << "Error: randomAttr null!\n";
      continue;
    }

    tensorType.setCacheData(llvm::dyn_cast<mlir::DenseElementsAttr>(randomAttr));

    inputs->addInput(inputName, arg);
    
    auto denseAttr = llvm::dyn_cast<mlir::DenseElementsAttr>(randomAttr);
  }

  llvm::dbgs() << "Info: created " << inputs->getInputCount()
               << " input tensors in createIoDef.\n";

  // Handle function return values (outputs)
  if (outputs) {
    auto funcType = targetFuncOp.getFunctionType();
    unsigned resultCount = funcType.getNumResults();
    
    if (resultCount == 0) {
      llvm::dbgs() << "Info: function has no return values\n";
    } else {
      llvm::dbgs() << "Info: function has " << resultCount << " return value(s)\n";
      
      for (unsigned i = 0; i < resultCount; ++i) {
        auto resultType = funcType.getResult(i);
        std::string outputName = "output_" + std::to_string(i);
        
        llvm::dbgs() << "Processing " << outputName << " result type: " << resultType << "\n";
        
        auto tensorType = llvm::dyn_cast<atir::TensorType>(resultType);
        if (!tensorType) {
          llvm::dbgs() << "Warning: result type is not atir::TensorType, skipping\n";
          continue;
        }
        
        // Create empty cache data for outputs (will be filled during execution)
        mlir::ArrayRef<int64_t> shape = tensorType.getShape();
        mlir::Type elementType = tensorType.getElementType();
        
        int64_t numElements = 1;
        std::vector<int64_t> staticShape;
        bool hasDynamicShape = false;
        for (int64_t dim : shape) {
          if (dim == mlir::ShapedType::kDynamic) {
            llvm::dbgs() << "Warning: dynamic shape in output, using DEFAULT_INPUT_DIM\n";
            dim = DEFAULT_INPUT_DIM;
            hasDynamicShape = true;
          }
          numElements *= dim;
          staticShape.push_back(dim);
        }
        
        if (staticShape.empty() || numElements <= 0) {
          llvm::dbgs() << "Warning: invalid output shape, skipping\n";
          continue;
        }
        
        mlir::RankedTensorType standardTensorType =
            mlir::RankedTensorType::get(staticShape, elementType);
        
        // Create zero-initialized DenseElementsAttr
        mlir::Attribute emptyAttr;
        if (elementType.isF32()) {
          std::vector<float> zeros(numElements, 0.0f);
          emptyAttr = mlir::DenseElementsAttr::get(standardTensorType, 
                                                    llvm::ArrayRef<float>(zeros.data(), zeros.size()));
        } else if (elementType.isF64()) {
          std::vector<double> zeros(numElements, 0.0);
          emptyAttr = mlir::DenseElementsAttr::get(standardTensorType, 
                                                    llvm::ArrayRef<double>(zeros.data(), zeros.size()));
        } else if (elementType.isInteger(32)) {
          std::vector<int32_t> zeros(numElements, 0);
          emptyAttr = mlir::DenseElementsAttr::get(standardTensorType, 
                                                    llvm::ArrayRef<int32_t>(zeros.data(), zeros.size()));
        } else if (elementType.isInteger(64) || mlir::isa<mlir::IndexType>(elementType)) {
          std::vector<int64_t> zeros(numElements, 0);
          emptyAttr = mlir::DenseElementsAttr::get(standardTensorType, 
                                                    llvm::ArrayRef<int64_t>(zeros.data(), zeros.size()));
        } else {
          llvm::errs() << "Warning: unsupported output element type: " << elementType << "\n";
          continue;
        }
        
        if (emptyAttr) {
          tensorType.setCacheData(llvm::dyn_cast<mlir::DenseElementsAttr>(emptyAttr));
          // For outputs, we need to preserve the original type information
          // We create a buffer operation to hold the output value with correct type
          mlir::OpBuilder builder(targetFuncOp->getContext());
          builder.setInsertionPointToStart(&targetFuncOp->getRegions().front().front());
          
          auto bufferOp = builder.create<atir::BufferOp>(
              targetFuncOp.getLoc(),
              atir::TensorType::get(staticShape, elementType, tensorType.getName()));
          
          outputs->addOutput(outputName, bufferOp.getResult());
        }
      }
    }
    
    llvm::dbgs() << "Info: created " << outputs->getOutputCount()
                 << " output tensors in createIoDef.\n\n";
  }
}

} // namespace atir
