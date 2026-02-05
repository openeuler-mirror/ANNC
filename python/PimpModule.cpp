#include "Dialect/Pimp/Passes/Passes.h"
#include "Conversion/Passes.h"
#include "Builder/Builder.h"
#include "PimpModule.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_kpgemm, m) {
  pimp::populateAttributeModule(m);
  pimp::populateTypesModule(m);

  pimp::registerPimpBlockFusionPass();
  pimp::registerPimpDistributePass();
  pimp::registerPimpEltwiseFusionPass();
  pimp::registerPimpTilingPass();
  pimp::registerPimpUnrollPass();
  pimp::registerPimpFastCodegenPass();
  pimp::registerPimpCanonicalizePass();
  pimp::registerPimpOpFusionPass();
  pimp::registerConvertPimpToAffinePass();
  pimp::registerConvertPimpToLinalgPass();

  auto pimp_m = m.def_submodule("pimp");
  pimp_m.def(
    "register_dialect",
    [](MlirContext context, bool load) {
      MlirDialectHandle handle = mlirGetDialectHandle__pimp__();
      mlirDialectHandleRegisterDialect(handle, context);
      if (load) {
        mlirDialectHandleLoadDialect(handle, context);
      }
    },
    py::arg("context") = py::none(), py::arg("load") = true);
}