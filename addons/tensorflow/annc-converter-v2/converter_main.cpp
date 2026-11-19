// CLI wrapper around the converter-v2 rewrite core (rewriteGraphDefInPlace).
// The core is file-IO free on purpose (shared with the PRELOAD plugin); this
// executable adds only argument parsing and GraphDef (de)serialization for
// the offline pipelines (annc-tf-pipeline-v2 and ad-hoc rewrites).
// Logging goes through the header-only Support/Log.h: errors at ERROR
// (visible at the default level), the verbose summary at INFO.
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Support/Log.h"
#include "graph_rewriter_core.h"
#include "tensorflow/core/framework/graph.pb.h"

namespace {

bool hasArg(int argc, char **argv, const std::string &name) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == name) return true;
  }
  return false;
}

std::string takeValue(int argc, char **argv, const std::string &name,
                      const std::string &defaultValue = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) return argv[i + 1];
  }
  return defaultValue;
}

bool readWholeFile(const std::string &path, std::string *out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    ANNC_LOG_ERROR("annc-converter-v2") << "cannot open " << path << "\n";
    return false;
  }
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  if (in.bad()) {
    ANNC_LOG_ERROR("annc-converter-v2")
        << "error while reading " << path << "\n";
    return false;
  }
  *out = std::move(text);
  return true;
}

bool writeWholeFile(const std::string &path, const std::string &content) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    ANNC_LOG_ERROR("annc-converter-v2")
        << "cannot open " << path << " for writing\n";
    return false;
  }
  out.write(content.data(), static_cast<std::streamsize>(content.size()));
  return static_cast<bool>(out);
}

void printUsage() {
  std::cout
      << "Usage: annc-converter-v2 --metadata_json <path> "
         "--input_graphdef <path> --output_graphdef <path>\n"
         "                          [--shared_lib_path <path>] "
         "[--atir_module_path <path>]\n"
         "                          [--kernel_name <name>] [--verbose]\n"
         "  Both paths may be given; each site kind consumes its own (AOT\n"
         "  sites read the shared library, JIT sites read the module).\n"
         "  --verbose maps onto ANNC_LOG_LEVEL=info (an explicit variable\n"
         "  wins).\n";
}

}  // namespace

int main(int argc, char **argv) {
  if (hasArg(argc, argv, "--help") || hasArg(argc, argv, "-h")) {
    printUsage();
    return 0;
  }
  const bool verbose = hasArg(argc, argv, "--verbose");
  // Map --verbose onto the logging level before the first log call (the
  // level is cached on first use); an explicit variable wins.
  if (verbose && !std::getenv("ANNC_LOG_LEVEL"))
    ::setenv("ANNC_LOG_LEVEL", "info", /*overwrite=*/0);

  const std::string metadataJson = takeValue(argc, argv, "--metadata_json");
  const std::string inputGraphDef = takeValue(argc, argv, "--input_graphdef");
  const std::string outputGraphDef =
      takeValue(argc, argv, "--output_graphdef");
  const std::string sharedLibPath = takeValue(argc, argv, "--shared_lib_path");
  const std::string atirModulePath = takeValue(argc, argv, "--atir_module_path");
  const std::string kernelName = takeValue(argc, argv, "--kernel_name");

  if (metadataJson.empty() || inputGraphDef.empty() ||
      outputGraphDef.empty()) {
    ANNC_LOG_ERROR("annc-converter-v2")
        << "--metadata_json, --input_graphdef and --output_graphdef are "
           "required\n";
    return 1;
  }

  std::string pbBytes;
  if (!readWholeFile(inputGraphDef, &pbBytes)) return 1;
  tensorflow::GraphDef graph;
  if (!graph.ParseFromString(pbBytes) || graph.node_size() == 0) {
    ANNC_LOG_ERROR("annc-converter-v2")
        << "failed to parse GraphDef (or empty): " << inputGraphDef << "\n";
    return 1;
  }

  auto metadata = annc::converter_v2::readFusionMetadataJson(metadataJson);
  if (!metadata.ok()) {
    ANNC_LOG_ERROR("annc-converter-v2") << metadata.error << "\n";
    return 1;
  }

  annc::converter_v2::RewritePaths paths;
  paths.sharedLibPath = sharedLibPath;
  paths.atirModulePath = atirModulePath;
  paths.kernelNameOverride = kernelName;
  paths.verbose = verbose;
  std::string error;
  if (!annc::converter_v2::rewriteGraphDefInPlace(&graph,
                                                  std::move(metadata.value),
                                                  paths, &error)) {
    ANNC_LOG_ERROR("annc-converter-v2") << error << "\n";
    return 1;
  }

  std::string serialized;
  if (!graph.SerializeToString(&serialized)) {
    ANNC_LOG_ERROR("annc-converter-v2") << "GraphDef serialization failed\n";
    return 1;
  }
  if (!writeWholeFile(outputGraphDef, serialized)) return 1;
  ANNC_LOG_INFO("annc-converter-v2")
      << "rewritten GraphDef written to: " << outputGraphDef << "\n";
  return 0;
}
