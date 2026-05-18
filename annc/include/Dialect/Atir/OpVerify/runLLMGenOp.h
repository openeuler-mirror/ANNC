#ifndef DIALECT_ATIR_OPVERIFY_RUNLLMGENOP_H
#define DIALECT_ATIR_OPVERIFY_RUNLLMGENOP_H

#include "Dialect/Atir/OpVerify/IoDef.h"
#include "Dialect/Atir/OpVerify/PerformanceMetrics.h"
#include <string>

namespace atir {

/// Run LLM generated operator via TensorFlow C++ API
/// @param libPath Path to the TensorFlow operator library
/// @param inputs Input tensor definitions
/// @param outputs Output tensor definitions
/// @param metrics Performance metrics output (only execution time)
/// @param config Warmup and profiling configuration
bool runLLMGenOp(const std::string &libPath, IoTensorDef *inputs, 
                 IoTensorDef *outputs,
                 PerformanceMetrics &metrics,
                 const WarmupConfig &config);

} // namespace atir

#endif // DIALECT_ATIR_OPVERIFY_RUNLLMGENOP_H
