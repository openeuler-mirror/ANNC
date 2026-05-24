#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "Dialect/Atir/AtirOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "nlohmann/json.hpp"

using namespace mlir;

namespace {

static std::string stringAttr(DictionaryAttr dict, StringRef key) {
  if (!dict) return "";
  Attribute raw = dict.get(key);
  if (auto attr = raw ? dyn_cast<StringAttr>(raw) : StringAttr()) return attr.str();
  return "";
}

static int64_t intAttr(DictionaryAttr dict, StringRef key,
                       int64_t defaultValue = 0) {
  if (!dict) return defaultValue;
  Attribute raw = dict.get(key);
  if (auto attr = raw ? dyn_cast<IntegerAttr>(raw) : IntegerAttr()) {
    return attr.getInt();
  }
  return defaultValue;
}

static std::vector<std::string> stringArrayAttr(DictionaryAttr dict,
                                                StringRef key) {
  std::vector<std::string> values;
  if (!dict) return values;
  Attribute raw = dict.get(key);
  auto arr = raw ? dyn_cast<ArrayAttr>(raw) : ArrayAttr();
  if (!arr) return values;
  for (Attribute attr : arr) {
    if (auto str = dyn_cast<StringAttr>(attr)) values.push_back(str.str());
  }
  return values;
}

static std::vector<int64_t> intArrayAttr(DictionaryAttr dict, StringRef key) {
  std::vector<int64_t> values;
  if (!dict) return values;
  Attribute raw = dict.get(key);
  auto arr = raw ? dyn_cast<ArrayAttr>(raw) : ArrayAttr();
  if (!arr) return values;
  for (Attribute attr : arr) {
    if (auto i = dyn_cast<IntegerAttr>(attr)) values.push_back(i.getInt());
  }
  return values;
}

static std::vector<int64_t> parseShapeString(const std::string &shape) {
  std::vector<int64_t> dims;
  std::stringstream ss(shape);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty() || item == "?") {
      dims.push_back(-1);
      continue;
    }
    dims.push_back(std::stoll(item));
  }
  return dims;
}

static std::vector<std::vector<int64_t>>
parseShapeStrings(const std::vector<std::string> &shapes) {
  std::vector<std::vector<int64_t>> parsed;
  for (const auto &shape : shapes) parsed.push_back(parseShapeString(shape));
  return parsed;
}

static std::optional<nlohmann::json> findANNCFusion(ModuleOp module) {
  std::optional<nlohmann::json> result;
  module.walk([&](atir::CustomizeOp op) {
    if (result.has_value() || op.getOpType() != "ANNCFused") return;
    auto metadata = op->getAttrOfType<DictionaryAttr>("metadata");
    if (!metadata) return;

    nlohmann::json info;
    info["name"] = stringAttr(metadata, "tf.name");
    info["pattern"] = stringAttr(metadata, "fusion.pattern");
    info["kernel_name"] = stringAttr(metadata, "kernel_name");
    info["output_tensor"] = stringAttr(metadata, "tf.output");
    info["original_nodes"] = stringArrayAttr(metadata, "tf.nodes");
    info["inputs"] = stringArrayAttr(metadata, "tf.inputs");
    info["input_shapes"] =
        parseShapeStrings(stringArrayAttr(metadata, "tf.input_shapes"));
    info["output_shape"] =
        parseShapeString(stringAttr(metadata, "tf.output_shape"));
    std::string abi = stringAttr(metadata, "abi");
    info["abi"] = abi.empty() ? "mlir_ciface" : abi;
    info["n_constants"] = intAttr(metadata, "Nconstants");
    info["n_fixed"] = intAttr(metadata, "Nfixed");
    info["n_dynamic"] = intAttr(metadata, "Ndynamic");
    info["num_outputs"] = intAttr(metadata, "num_outputs", 1);
    info["output_ranks"] = intArrayAttr(metadata, "output_ranks");
    info["input_ranks"] = intArrayAttr(metadata, "input_ranks");
    info["dynamic_dims"] = intArrayAttr(metadata, "dynamic_dims");
    info["kernel_arg_order"] = intArrayAttr(metadata, "kernel_arg_order");
    info["symbolic_signature"] = stringAttr(metadata, "symbolic_signature");
    info["fallback_function"] = stringAttr(metadata, "fallback_function");

    if (!info["name"].get<std::string>().empty() &&
        !info["original_nodes"].empty() && !info["inputs"].empty() &&
        !info["output_tensor"].get<std::string>().empty()) {
      result = std::move(info);
    }
  });
  return result;
}

}  // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<fused atir .mlir>"),
      llvm::cl::Required);
  llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output metadata JSON"),
      llvm::cl::value_desc("filename"), llvm::cl::Required);

  llvm::cl::ParseCommandLineOptions(
      argc, argv, "annc-fusion-metadata: extract ANNCFused metadata\n");

  DialectRegistry registry;
  registry.insert<func::FuncDialect, atir::AtirDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = parseSourceFile<ModuleOp>(inputFilename, &context);
  if (!module) {
    llvm::errs() << "[annc-fusion-metadata] Error: failed to parse ATIR: "
                 << inputFilename << "\n";
    return 1;
  }

  auto fusion = findANNCFusion(*module);
  if (!fusion.has_value()) {
    llvm::errs() << "[annc-fusion-metadata] Error: no ANNCFused metadata in "
                 << inputFilename << "\n";
    return 1;
  }

  std::ofstream out(outputFilename);
  if (!out.is_open()) {
    llvm::errs() << "[annc-fusion-metadata] Error: cannot open output: "
                 << outputFilename << "\n";
    return 1;
  }
  out << fusion->dump(2) << "\n";
  return 0;
}
