#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "FusionMetadata/FusionMetadata.h"
#include "FusionMetadata/FusionMetadataJson.h"
#include "GraphDefRewriter.h"

#include "llvm/Support/Error.h"

namespace {

struct ConverterOptions {
  std::string fusedAtirPath;
  std::string inputGraphDef;
  std::string outputGraphDef;
  std::string kernelName;
  std::string sharedLibPath;
  std::string metadataJson;
  bool verbose = false;
};

void printUsage(std::ostream &stream) {
  stream << "Usage: annc-converter [fused-atir.mlir] "
            "--input_graphdef <path> --output_graphdef <path> "
            "--shared_lib_path <path> [--metadata_json <path>] "
            "[--kernel_name <name>] [--verbose]\n";
}

bool takeOptionValue(int argc, char **argv, int *index, std::string *value,
                     const char *option) {
  if (*index + 1 >= argc) {
    std::cerr << "[annc-converter] Error: " << option << " requires a value\n";
    return false;
  }
  *value = argv[++*index];
  return true;
}

bool parseOptions(int argc, char **argv, ConverterOptions *options,
                  bool *showHelp) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    const size_t equalPos = arg.find('=');
    const bool hasAttachedValue =
        arg.rfind("--", 0) == 0 && equalPos != std::string::npos;
    const std::string option =
        hasAttachedValue ? arg.substr(0, equalPos) : arg;
    auto takeValue = [&](std::string *value, const char *optionName) {
      if (hasAttachedValue) {
        *value = arg.substr(equalPos + 1);
        return true;
      }
      return takeOptionValue(argc, argv, &i, value, optionName);
    };

    if (option == "--help" || option == "-h") {
      *showHelp = true;
      return true;
    }
    if (option == "--tf-graphdef-rewrite") {
      continue;
    }
    if (option == "--verbose") {
      options->verbose = true;
    } else if (option == "--input_graphdef") {
      if (!takeValue(&options->inputGraphDef, "--input_graphdef")) {
        return false;
      }
    } else if (option == "--output_graphdef") {
      if (!takeValue(&options->outputGraphDef, "--output_graphdef")) {
        return false;
      }
    } else if (option == "--shared_lib_path") {
      if (!takeValue(&options->sharedLibPath, "--shared_lib_path")) {
        return false;
      }
    } else if (option == "--metadata_json") {
      if (!takeValue(&options->metadataJson, "--metadata_json")) {
        return false;
      }
    } else if (option == "--kernel_name") {
      if (!takeValue(&options->kernelName, "--kernel_name")) {
        return false;
      }
    } else if (!arg.empty() && arg[0] != '-' &&
               options->fusedAtirPath.empty()) {
      options->fusedAtirPath = std::move(arg);
    } else {
      std::cerr << "[annc-converter] Error: unknown option: " << arg << "\n";
      return false;
    }
  }
  return true;
}

void reportError(llvm::Error error) {
  llvm::handleAllErrors(
      std::move(error), [](const llvm::ErrorInfoBase &errorInfo) {
        std::cerr << "[annc-converter] Error: " << errorInfo.message() << "\n";
      });
}

}  // namespace

int main(int argc, char **argv) {
  ConverterOptions options;
  bool showHelp = false;
  if (!parseOptions(argc, argv, &options, &showHelp)) {
    printUsage(std::cerr);
    return 1;
  }
  if (showHelp) {
    printUsage(std::cout);
    return 0;
  }

  if (options.inputGraphDef.empty() || options.outputGraphDef.empty()) {
    std::cerr << "[annc-converter] Error: --input_graphdef and "
                 "--output_graphdef are required\n";
    return 1;
  }
  if (options.sharedLibPath.empty()) {
    std::cerr << "[annc-converter] Error: --shared_lib_path is required\n";
    return 1;
  }
  if (options.metadataJson.empty() && options.fusedAtirPath.empty()) {
    std::cerr << "[annc-converter] Error: provide fused ATIR input or "
                 "--metadata_json\n";
    return 1;
  }

  std::vector<annc::fusion::FusionInfo> fusionInfos;
  if (!options.metadataJson.empty()) {
    auto fusions = annc::fusion::readFusionMetadataJson(options.metadataJson);
    if (!fusions) {
      reportError(fusions.takeError());
      return 1;
    }
    fusionInfos = std::move(*fusions);
  } else {
    auto fusions =
        annc::fusion::extractFusionInfosFromMlirText(options.fusedAtirPath);
    if (!fusions) {
      reportError(fusions.takeError());
      return 1;
    }
    fusionInfos = std::move(*fusions);
  }

  annc::fusion::GraphDefRewriteOptions rewriteOptions;
  rewriteOptions.inputGraphPath = options.inputGraphDef;
  rewriteOptions.outputGraphPath = options.outputGraphDef;
  rewriteOptions.kernelNameOverride = options.kernelName;
  rewriteOptions.sharedLibPath = options.sharedLibPath;
  rewriteOptions.verbose = options.verbose;

  return annc::fusion::rewriteGraphDefWithANNCFused(std::move(fusionInfos),
                                                    rewriteOptions)
             ? 0
             : 1;
}
