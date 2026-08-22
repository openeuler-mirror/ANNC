#include "Dialect/Atir/TemplateFingerprint.h"

#include "Dialect/Atir/AtirOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"

namespace atir {
namespace {

constexpr llvm::StringLiteral kTemplateFingerprintSchema =
    "annc-atir-template-v2";
constexpr llvm::StringLiteral kIntraThreadCountAttr =
    "annc.intra_thread_count";

mlir::Type canonicalType(mlir::Type type) {
  if (auto tensor = mlir::dyn_cast<TensorType>(type)) {
    llvm::SmallVector<int64_t> dynamicShape(tensor.getShape().size(),
                                            mlir::ShapedType::kDynamic);
    return TensorType::get(dynamicShape, tensor.getElementType(), {},
                           tensor.getEncoding(), tensor.getStride(),
                           tensor.getLayout(), tensor.getMemType(),
                           tensor.getAddress(), tensor.getDeviceParallel(),
                           tensor.getOnchipParallel(), tensor.getCacheData());
  }
  if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    llvm::SmallVector<int64_t> dynamicShape(tensor.getShape().size(),
                                            mlir::ShapedType::kDynamic);
    return mlir::RankedTensorType::get(dynamicShape, tensor.getElementType(),
                                       tensor.getEncoding());
  }
  return type;
}

void canonicalizeValueTypes(mlir::Operation *root) {
  root->walk([](mlir::Operation *operation) {
    for (mlir::Region &region : operation->getRegions()) {
      for (mlir::Block &block : region) {
        for (mlir::BlockArgument argument : block.getArguments()) {
          argument.setType(canonicalType(argument.getType()));
        }
      }
    }
    for (mlir::OpResult result : operation->getResults()) {
      result.setType(canonicalType(result.getType()));
    }
  });
}

void removeSourceIdentity(mlir::Operation *root) {
  root->walk([](mlir::Operation *operation) {
    operation->removeAttr("tf.name");
    operation->removeAttr("tf_name");

    auto metadata = operation->getAttrOfType<mlir::DictionaryAttr>("metadata");
    if (!metadata) return;
    llvm::SmallVector<mlir::NamedAttribute> canonical;
    for (mlir::NamedAttribute attribute : metadata) {
      llvm::StringRef name = attribute.getName().strref();
      if (name == "tf.name" || name == "tf_name") continue;
      canonical.push_back(attribute);
    }
    if (canonical.empty()) {
      operation->removeAttr("metadata");
    } else if (canonical.size() != metadata.size()) {
      operation->setAttr("metadata", mlir::DictionaryAttr::get(
                                         operation->getContext(), canonical));
    }
  });
}

}  // namespace

std::string computeAtirTemplateFingerprint(mlir::ModuleOp module,
                                           mlir::func::FuncOp function) {
  mlir::OwningOpRef<mlir::Operation *> owner(function->clone());
  auto clone = mlir::cast<mlir::func::FuncOp>(owner.get());

  clone.setName("__annc_kernel_template");
  clone->removeAttr("fusion.metadata");
  removeSourceIdentity(clone);
  canonicalizeValueTypes(clone);

  llvm::SmallVector<mlir::Type> argumentTypes;
  argumentTypes.reserve(clone.getNumArguments());
  for (mlir::BlockArgument argument : clone.getArguments()) {
    argumentTypes.push_back(argument.getType());
  }
  clone.setType(mlir::FunctionType::get(clone.getContext(), argumentTypes,
                                        clone.getResultTypes()));

  std::string canonical;
  llvm::raw_string_ostream output(canonical);
  output << "schema=" << kTemplateFingerprintSchema << '\n';
  output << kIntraThreadCountAttr << '=';
  auto intraThreadCount = module->getAttrOfType<mlir::IntegerAttr>(
      kIntraThreadCountAttr);
  // The lowering pass treats an omitted attribute as serial execution.
  output << (intraThreadCount ? intraThreadCount.getInt() : 1) << '\n';

  mlir::OpPrintingFlags flags;
  flags.printGenericOpForm().useLocalScope();
  clone->print(output, flags);
  output.flush();

  llvm::SHA256 sha;
  sha.update(canonical);
  auto digest = sha.final();
  return llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true);
}

}  // namespace atir
