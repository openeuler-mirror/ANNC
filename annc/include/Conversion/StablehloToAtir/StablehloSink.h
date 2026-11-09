#ifndef ANNC_CONVERSION_STABLEHLO_TO_ATIR_SINK_H
#define ANNC_CONVERSION_STABLEHLO_TO_ATIR_SINK_H

#include <map>
#include <memory>
#include <string>

namespace mlir {
class DialectRegistry;
class Pass;
class Type;
}  // namespace mlir

namespace annc {

// [residue-sink] stat report (ordered maps keep output deterministic).
struct SinkStat {
  std::map<std::string, int> residueMapped;    // tf op type -> count
  std::map<std::string, int> residueFallback;  // op type -> count
  std::map<std::string, int> residueFallbackReason;  // reason -> count
  std::map<std::string, int> keep;             // kept op name -> count
};

// Renders the stat report as `[residue-sink] stat: ...` lines on stderr.
void printSinkStat(const SinkStat& stat);

namespace tfcompat {

// Minimal "tf_type" compat dialect: !tf_type.string parses losslessly.
void registerTfTypeCompatDialect(mlir::DialectRegistry& registry);

bool isTfStringType(mlir::Type type);

}  // namespace tfcompat

// Creates the aggregate residue-sink pass (TableGen pass base materializes
// in the .cpp where `using namespace mlir` is in scope for ModuleOp).
std::unique_ptr<mlir::Pass> createStablehloSink();

}  // namespace annc

#endif  // ANNC_CONVERSION_STABLEHLO_TO_ATIR_SINK_H
