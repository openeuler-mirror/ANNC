#include "Dialect/Atir/Passes/Passes.h"
#include "Conversion/Passes.h"
#include "Builder/Builder.h"
#include "AtirModule.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_annc, m) {
  atir::populateAttributeModule(m);
  atir::populateTypesModule(m);

  atir::registerAtirBlockFusionPass();
  atir::registerAtirDistributePass();
  atir::registerAtirEltwiseFusionPass();
  atir::registerAtirTilingPass();
  atir::registerAtirUnrollPass();
  atir::registerAtirFastCodegenPass();
  atir::registerAtirCanonicalizePass();
  atir::registerAtirOpFusionPass();
  atir::registerConvertAtirToAffinePass();
  atir::registerConvertAtirToLinalgPass();

  auto atir_m = m.def_submodule("atir");
  atir_m.def(
    "register_dialect",
    [](MlirContext context, bool load) {
      MlirDialectHandle handle = mlirGetDialectHandle__atir__();
      mlirDialectHandleRegisterDialect(handle, context);
      if (load) {
        mlirDialectHandleLoadDialect(handle, context);
      }
    },
    py::arg("context") = py::none(), py::arg("load") = true);
}