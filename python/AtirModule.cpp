#include "Dialect/Atir/Passes/Passes.h"
#include "Dialect-c/Passes.h"
#include "Conversion/Passes.h"
#include "AtirModule.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_annc, m) {
  atir::populateAttributeModule(m);
  atir::populateTypesModule(m);

  atir::registerAnncFastCodegenPass();
  atir::registerAtirOpFusionPass();
  atir::registerConvertAtirToLinalgPass();
  atir::registerConvertAtirV2ToLinalgPass();
  atir::registerLowerContextKernelEntriesPass();
  atir::registerConvertContextKernelABIPass();
  anncRegisterAllTargetPasses();

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
