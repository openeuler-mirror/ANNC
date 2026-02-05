#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/Registration.h"
#include "Dialect-c/Dialects.h"
#include "Dialect/Pimp/PimpOps.h"
#include <pybind11/numpy.h>

namespace py = pybind11;
namespace pimp {
void populateAttributeModule(py::module_ m);
void populateTypesModule(py::module_ m);
} // namespace pimp