#ifndef DIALECT_ATIR_OPVERIFY_RUNKPGENOP_H
#define DIALECT_ATIR_OPVERIFY_RUNKPGENOP_H

#include "Dialect/Atir/OpVerify/IoDef.h"
#include "Dialect/Atir/OpVerify/PerformanceMetrics.h"
#include <string>
#include <optional>

namespace atir {

/// Run KP (Kernel Provider) generated operator with performance metrics
/// @param libPath Path to the kernel library
/// @param inputs Input tensor definitions
/// @param outputTemplate Output template (from createIoDef, contains return value definitions)
/// @param outputResult Output results (will be filled with execution results)
/// @param metrics Performance metrics output
/// @param config Warmup and profiling configuration
bool runKpGenOp(const std::string &libPath, 
                const IoTensorDef *inputs,
                const IoTensorDef *outputTemplate,
                IoTensorDef *outputResult,
                PerformanceMetrics &metrics,
                const WarmupConfig &config);

} // namespace atir

#endif // DIALECT_ATIR_OPVERIFY_RUNKPGENOP_H
