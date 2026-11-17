#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "Dialect/Atir/AtirOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "nlohmann/json.hpp"
#include "onnx/common/file_utils.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx_preprocess.h"
#include "src/Builder/FrontendDialectTransformer.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace onnx_mlir;

void registerOnnxStringToFrameworkCustomCallPass();
void registerAnncFrameworkDialect(DialectRegistry& registry);
void registerFrameworkVerifyPass();
void registerFrameworkShapePass();
void registerFrameworkCustomCallToStablehloPass();
void registerOnnxSourceFinalizePass();

namespace {
using Json = nlohmann::json;

Json inferTensorTypes(const onnx::ModelProto& model) {
  onnx::ModelProto inferred = model;
  try {
    onnx::shape_inference::InferShapes(
        inferred, onnx::OpSchemaRegistry::Instance(),
        onnx::ShapeInferenceOptions(false, 0, false));
  } catch (const std::runtime_error&) {
    // Match the previous ONNX query: inference failure leaves declarations
    // available, without publishing partially inferred types.
    inferred = model;
  }
  Json types = Json::object();
  auto add = [&](const auto& values) {
    for (const auto& value : values) {
      if (!value.type().has_tensor_type()) continue;
      const auto& tensor = value.type().tensor_type();
      if (!tensor.elem_type()) continue;
      Json info{{"elem_type", tensor.elem_type()}};
      if (tensor.has_shape()) {
        info["shape"] = Json::array();
        for (const auto& dim : tensor.shape().dim())
          info["shape"].push_back(dim.has_dim_value() ? Json(dim.dim_value())
                                                      : Json(nullptr));
      }
      types[value.name()] = std::move(info);
    }
  };
  add(inferred.graph().value_info());
  add(inferred.graph().input());
  add(inferred.graph().output());
  for (const auto& tensor : inferred.graph().initializer())
    types[tensor.name()] = {
        {"elem_type", tensor.data_type()},
        {"shape",
         std::vector<int64_t>(tensor.dims().begin(), tensor.dims().end())}};
  return types;
}

Json buildModelIndex(const onnx::ModelProto& model) {
  if (!model.has_graph()) throw std::runtime_error("ONNX model has no graph");
  const auto& graph = model.graph();
  if (model.functions_size() || model.training_info_size() ||
      graph.sparse_initializer_size())
    throw std::runtime_error(
        "local functions, training graphs and sparse initializers unsupported");
  for (const auto& node : graph.node())
    for (const auto& attr : node.attribute())
      if (attr.has_g() || attr.graphs_size() || attr.has_sparse_tensor() ||
          attr.sparse_tensors_size())
        throw std::runtime_error(
            "subgraphs and sparse tensor attributes unsupported");

  Json tensors = Json::array(), nodes = Json::array();
  std::map<std::string, size_t> positions;
  auto define = [&](const std::string& name, const char* field, Json value) {
    if (name.empty())
      throw std::runtime_error(
          "empty graph input, output, or initializer name");
    auto [it, inserted] = positions.emplace(name, tensors.size());
    if (inserted)
      tensors.push_back({{"name", name}, {"consumers", Json::array()}});
    auto& record = tensors[it->second];
    if (record.contains(field))
      throw std::runtime_error("multiple " + std::string(field) +
                               " definitions for tensor '" + name + "'");
    record[field] = std::move(value);
  };
  for (int slot = 0; slot != graph.input_size(); ++slot)
    define(graph.input(slot).name(), "graph_input", slot);
  for (const auto& tensor : graph.initializer())
    define(tensor.name(), "initializer", true);
  std::set<std::string> names;
  for (const auto& node : graph.node()) {
    const std::string& name = node.name();
    if (name.empty()) throw std::runtime_error("ONNX node has an empty name");
    if (!names.insert(name).second)
      throw std::runtime_error("duplicate ONNX node name: '" + name + "'");
    bool namedOutput = false;
    for (const auto& output : node.output()) namedOutput |= !output.empty();
    if (node.op_type().empty() || !namedOutput)
      throw std::runtime_error("node " + name +
                               " has no operation or named output");
    nodes.push_back(
        {{"name", name},
         {"domain", node.domain()},
         {"op_type", node.op_type()},
         {"inputs",
          std::vector<std::string>(node.input().begin(), node.input().end())},
         {"outputs", std::vector<std::string>(node.output().begin(),
                                              node.output().end())}});
    for (int slot = 0; slot != node.input_size(); ++slot) {
      const auto& input = node.input(slot);
      if (input.empty()) continue;
      auto found = positions.find(input);
      if (found == positions.end())
        throw std::runtime_error("undefined input '" + input + "' at " + name);
      tensors[found->second]["consumers"].push_back(
          {{"node", name}});
    }
    for (int slot = 0; slot != node.output_size(); ++slot) {
      const auto& output = node.output(slot);
      if (output.empty()) continue;
      if (positions.count(output))
        throw std::runtime_error("multiple definitions for tensor '" + output +
                                 "'");
      define(output, "producer", {{"node", name}});
    }
  }
  for (int slot = 0; slot != graph.output_size(); ++slot) {
    const auto& name = graph.output(slot).name();
    auto found = positions.find(name);
    if (found == positions.end())
      throw std::runtime_error("undefined graph output '" + name + "'");
    auto& tensor = tensors[found->second];
    if (!tensor.contains("graph_outputs"))
      tensor["graph_outputs"] = Json::array();
    tensor["graph_outputs"].push_back(slot);
  }
  const Json types = inferTensorTypes(model);
  for (auto& tensor : tensors) {
    auto type = types.find(tensor.at("name").get<std::string>());
    if (type != types.end()) tensor["type"] = *type;
  }
  return {{"schema", "annc-model-index/2"},
          {"graph",
           {{"nodes", std::move(nodes)},
            {"tensors", std::move(tensors)}}}};
}

llvm::cl::opt<bool> importOnnx(
    "annc-import-onnx",
    llvm::cl::desc("Import ONNX and retain its original graph for binding"));
}  // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  removeUnrelatedOptions({&OnnxMlirCommonOptions, &OnnxMlirOptOptions});

  registerOnnxSlicePreprocessPass();
  registerOnnxDynamicTilePreprocessPass();
  registerOnnxStringToFrameworkCustomCallPass();
  registerFrameworkVerifyPass();
  registerFrameworkShapePass();
  registerFrameworkCustomCallToStablehloPass();
  registerOnnxSourceFinalizePass();
  registerTransformsPasses();

  DialectRegistry registry;
  registry.insert<atir::AtirDialect, affine::AffineDialect, arith::ArithDialect,
                  func::FuncDialect, shape::ShapeDialect, tensor::TensorDialect,
                  stablehlo::StablehloDialect, ONNXDialect>();
  registerAnncFrameworkDialect(registry);

  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  PassPipelineCLParser passPipeline("", "Compiler passes to run");
  if (!llvm::cl::ParseCommandLineOptions(
          argc, argv, "ANNC ONNX framework passes\n", &llvm::errs()))
    return 1;

  std::string errorMessage;

  if (importOnnx) {
    try {
      // Load the protobuf once; both indexing and ONNX-MLIR import use it.
      std::error_code fileError;
      if (std::filesystem::hard_link_count(inputFilename, fileError) > 1 &&
          !fileError)
        throw std::runtime_error(
            "ONNX input is a hardlink, which ONNX-MLIR rejects");
      onnx::ModelProto model;
      onnx::LoadProtoFromPath(inputFilename, model);
      const auto index = buildModelIndex(model);
      MLIRContext context(registry);
      context.loadAllAvailableDialects();
      OwningOpRef<ModuleOp> module;
      ImportOptions options;
      // Match onnx-mlir's CLI defaults (ImportOptions itself defaults to
      // false). Capture the source index before ONNX inference mutates
      // declarations.
      options.useOnnxModelTypes = true;
      onnx::shape_inference::InferShapes(model);
      options.externalDataDir =
          std::filesystem::absolute(inputFilename).parent_path().string();
      ImportFrontendModel(model, context, module, options);
      if (!module) throw std::runtime_error("ONNX-MLIR import failed");
      (*module)->setAttr("annc.onnx_model",
                         StringAttr::get(&context, index.dump()));
      auto output = openOutputFile(outputBaseName, &errorMessage);
      if (!output) throw std::runtime_error(errorMessage);
      module->print(output->os());
      output->os() << "\n";
      output->keep();
      return 0;
    } catch (const std::exception& error) {
      llvm::errs() << "ONNX import: " << error.what() << "\n";
      return 1;
    }
  }

  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << "Failed to open input: " << errorMessage << "\n";
    return 1;
  }
  auto output = openOutputFile(outputBaseName, &errorMessage);
  if (!output) {
    llvm::errs() << "Failed to open output: " << errorMessage << "\n";
    return 1;
  }

  MlirOptMainConfig config;
  config.setPassPipelineParser(passPipeline)
      .splitInputFile(split_input_file ? kDefaultSplitMarker : "")
      .verifyDiagnostics(verify_diagnostics)
      .verifyPasses(verify_passes)
      .allowUnregisteredDialects(allowUnregisteredDialects)
      .emitBytecode(false)
      .useExplicitModule(false);
  if (failed(MlirOptMain(output->os(), std::move(input), registry, config)))
    return 1;

  output->keep();
  return 0;
}
