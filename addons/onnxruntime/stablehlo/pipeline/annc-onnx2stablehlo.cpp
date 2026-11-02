#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <system_error>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/raw_ostream.h"

#ifndef ANNC_LLVM_VERSION
#define ANNC_LLVM_VERSION "unknown"
#endif

namespace {

llvm::cl::opt<std::string> inputModel(llvm::cl::Positional,
                                      llvm::cl::desc("ONNX model"),
                                      llvm::cl::Required);
llvm::cl::opt<std::string> outputFile("o", llvm::cl::desc("StableHLO MLIR"),
                                      llvm::cl::value_desc("path"),
                                      llvm::cl::Required);
llvm::cl::opt<std::string> metadataFile(
    "metadata", llvm::cl::desc("Boundary metadata JSON"), llvm::cl::Required);
llvm::cl::opt<std::string> toolDir(
    "tool-dir",
    llvm::cl::desc("Directory containing external ONNX-MLIR tools"));
llvm::cl::opt<std::string> onnxMlir(
    "onnx-mlir", llvm::cl::desc("Path to the external ONNX-MLIR importer"));
llvm::cl::opt<std::string> onnxMlirOpt(
    "onnx-mlir-opt",
    llvm::cl::desc("Path to the external ONNX-MLIR optimizer"));
llvm::cl::opt<std::string> frameworkOpt(
    "framework-opt",
    llvm::cl::desc(
        "Path to the framework-only custom-call/residual optimizer"));
llvm::cl::opt<std::string> producerVersion(
    "producer-version",
    llvm::cl::desc("ONNX-MLIR producer version recorded in metadata"));
llvm::cl::opt<std::string> stablehloVersion(
    "stablehlo-version",
    llvm::cl::desc("StableHLO producer version recorded in metadata"));

std::string envOrUnknown(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && *value != '\0' ? value : "unknown";
}

std::string executablePath(llvm::StringRef explicitPath, llvm::StringRef name,
                           llvm::StringRef defaultDir) {
  if (!explicitPath.empty()) return explicitPath.str();
  if (!defaultDir.empty()) {
    llvm::SmallString<256> path(defaultDir);
    llvm::sys::path::append(path, name);
    if (llvm::sys::fs::exists(path)) return path.str().str();
  }
  return name.str();
}

int runStage(llvm::StringRef label, llvm::StringRef program,
             const std::vector<std::string> &arguments,
             const std::filesystem::path &work) {
  llvm::SmallVector<llvm::StringRef, 16> args;
  args.push_back(program);
  for (const std::string &argument : arguments) args.push_back(argument);
  const std::string stdoutPath = (work / "stage.stdout").string();
  const std::string stderrPath = (work / "stage.stderr").string();
  const std::optional<llvm::StringRef> redirects[] = {std::nullopt, stdoutPath,
                                                      stderrPath};
  // LLVM's redirection API may reuse existing files without truncation.
  {
    std::ofstream clear(stdoutPath, std::ios::trunc);
  }
  {
    std::ofstream clear(stderrPath, std::ios::trunc);
  }
  const int result =
      llvm::sys::ExecuteAndWait(program, args, std::nullopt, redirects);
  auto readText = [](const std::string &path) {
    std::ifstream input(path);
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
  };
  const std::string stdoutText = readText(stdoutPath);
  const std::string stderrText = readText(stderrPath);
  llvm::outs() << stdoutText;
  llvm::errs() << stderrText;
  if (result != 0) {
    llvm::errs() << "annc-onnx2stablehlo: " << label
                 << " failed with exit code " << result << "\n";
    llvm::json::Array command;
    for (llvm::StringRef argument : args) command.push_back(argument.str());
    llvm::json::Object failure{
        {"schema", "annc-stablehlo-boundary/failure/1"},
        {"source_model",
         std::filesystem::absolute(inputModel.getValue()).string()},
        {"stage", label.str()},
        {"exit_code", result},
        {"command", std::move(command)},
        {"diagnostics", stderrText},
        {"stdout", stdoutText}};
    std::error_code error;
    llvm::raw_fd_ostream stream(metadataFile.getValue() + ".failure.json",
                                error, llvm::sys::fs::OF_Text);
    if (error)
      llvm::errs() << "annc-onnx2stablehlo: cannot write failure record: "
                   << error.message() << "\n";
    else
      stream << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(failure)));
  }
  return result;
}

struct Counts {
  std::map<std::string, int64_t> dialects;
  std::map<std::string, int64_t> ops;
  std::map<std::string, int64_t> types;
};

Counts countText(llvm::StringRef text) {
  Counts counts;
  llvm::SmallVector<llvm::StringRef, 64> lines;
  text.split(lines, '\n');
  for (llvm::StringRef line : lines) {
    size_t begin = line.find('"');
    if (begin != llvm::StringRef::npos) {
      size_t end = line.find('"', begin + 1);
      if (end != llvm::StringRef::npos) {
        llvm::StringRef name = line.slice(begin + 1, end);
        size_t dot = name.find('.');
        if (dot != llvm::StringRef::npos &&
            line.substr(end + 1).starts_with("(")) {
          ++counts.dialects[name.substr(0, dot).str()];
          ++counts.ops[name.str()];
        }
      }
    }
    for (llvm::StringRef dialect :
         {llvm::StringRef("stablehlo."), llvm::StringRef("arith."),
          llvm::StringRef("shape.")}) {
      if (line.contains('"') && line.find(dialect) != llvm::StringRef::npos)
        continue;
      size_t pos = line.find(dialect);
      if (pos == llvm::StringRef::npos) continue;
      size_t end = pos;
      while (end < line.size() &&
             (std::isalnum(static_cast<unsigned char>(line[end])) ||
              line[end] == '_' || line[end] == '.'))
        ++end;
      llvm::StringRef name = line.slice(pos, end);
      size_t dot = name.find('.');
      ++counts.dialects[name.substr(0, dot).str()];
      ++counts.ops[name.str()];
    }
    if (line.contains("tensor<")) ++counts.types["tensor_lines"];
  }
  return counts;
}

std::string stripEntryPoint(llvm::StringRef text) {
  std::string result;
  llvm::SmallVector<llvm::StringRef, 64> lines;
  text.split(lines, '\n');
  for (llvm::StringRef line : lines) {
    if (line.contains("\"onnx.EntryPoint\"")) continue;
    result += line.str();
    result.push_back('\n');
  }
  return result;
}

llvm::json::Object jsonCounts(const std::map<std::string, int64_t> &counts) {
  llvm::json::Object result;
  for (const auto &[key, value] : counts) result[key] = value;
  return result;
}

llvm::json::Object stageRecord(llvm::StringRef program,
                               const std::vector<std::string> &arguments,
                               int exitCode) {
  llvm::json::Array command;
  command.push_back(program.str());
  for (const std::string &argument : arguments) command.push_back(argument);
  return llvm::json::Object{{"command", std::move(command)},
                            {"exit_code", exitCode}};
}

bool writeMetadata(
    const std::filesystem::path &path, const std::filesystem::path &model,
    const std::filesystem::path &output, const std::filesystem::path &imported,
    const std::filesystem::path &lowered, llvm::StringRef importer,
    llvm::StringRef optimizer, llvm::StringRef helper, llvm::StringRef producer,
    llvm::StringRef stablehlo, const std::vector<std::string> &importArgs,
    const std::vector<std::string> &preprocessArgs,
    const std::vector<std::string> &lowerArgs,
    const std::vector<std::string> &residualArgs, int importCode,
    int preprocessCode, int lowerCode, int residualCode, bool helperAvailable,
    const Counts &counts) {
  llvm::json::Object stages;
  stages["onnx_import"] = stageRecord(importer, importArgs, importCode);
    stages["preprocessing_passes"] =
      helperAvailable ? stageRecord(helper, preprocessArgs, preprocessCode)
                      : stageRecord("copy", {}, preprocessCode);
  stages["onnx_to_stablehlo"] = stageRecord(optimizer, lowerArgs, lowerCode);
  stages["residual_lowering"] =
      helperAvailable ? stageRecord(helper, residualArgs, residualCode)
                      : stageRecord("none", {}, residualCode);
  llvm::json::Object root{
      {"schema", "annc-stablehlo-boundary/1"},
      {"producer",
       llvm::json::Object{
           {"tool", "onnx-mlir"},
           {"version", producer.str()},
           {"stablehlo_version", stablehlo.str()},
           {"llvm_mlir_version", envOrUnknown("ANNC_ONNX_MLIR_LLVM_VERSION")}}},
      {"driver_llvm_version", ANNC_LLVM_VERSION},
      {"source_model", model.string()},
      {"stablehlo_file", output.string()},
      {"imported_onnx_mlir", imported.string()},
      {"lowered_intermediate", lowered.string()},
      {"stages", std::move(stages)},
      {"dialect_counts", jsonCounts(counts.dialects)},
      {"op_counts", jsonCounts(counts.ops)},
      {"type_counts", jsonCounts(counts.types)}};
  std::error_code error;
  llvm::raw_fd_ostream stream(path.string(), error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "annc-onnx2stablehlo: cannot write metadata " << path
                 << ": " << error.message() << "\n";
    return false;
  }
  stream << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(root)));
  stream.close();
  if (stream.has_error()) {
    llvm::errs() << "annc-onnx2stablehlo: cannot flush metadata " << path
                 << "\n";
    stream.clear_error();
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Standalone ONNX to StableHLO process boundary\n");

  const std::filesystem::path model =
      std::filesystem::absolute(inputModel.getValue());
  const std::filesystem::path output =
      std::filesystem::absolute(outputFile.getValue());
  const std::filesystem::path metadata =
      std::filesystem::absolute(metadataFile.getValue());
  if (!std::filesystem::is_regular_file(model)) {
    llvm::errs() << "annc-onnx2stablehlo: input model does not exist: " << model
                 << "\n";
    return 1;
  }
  if (!std::filesystem::is_directory(output.parent_path()) ||
      !std::filesystem::is_directory(metadata.parent_path())) {
    llvm::errs() << "annc-onnx2stablehlo: output directories must exist\n";
    return 1;
  }
  if (output == metadata || output == model || metadata == model ||
      (std::filesystem::exists(output) &&
       !std::filesystem::is_regular_file(output)) ||
      (std::filesystem::exists(metadata) &&
       !std::filesystem::is_regular_file(metadata))) {
    llvm::errs() << "annc-onnx2stablehlo: output and metadata must be distinct "
                    "regular file destinations\n";
    return 1;
  }

  llvm::SmallString<256> work;
  work = output.parent_path().string();
  llvm::sys::path::append(work, "annc-onnx2stablehlo-%%%%%%");
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory(work, work)) {
    llvm::errs() << "annc-onnx2stablehlo: cannot create temporary directory: "
                 << error.message() << "\n";
    return 1;
  }
  const std::filesystem::path temporary(work.str().str());
  auto cleanup = [&] {
    std::error_code ignored;
    std::filesystem::remove_all(temporary, ignored);
  };

  const std::filesystem::path executable = llvm::sys::fs::getMainExecutable(
      argv[0], reinterpret_cast<void *>(&main));
  const std::string tools =
      toolDir.empty() ? executable.parent_path().string() : toolDir.getValue();
  const std::string importer = executablePath(onnxMlir, "onnx-mlir", tools);
  const std::string optimizer =
      executablePath(onnxMlirOpt, "onnx-mlir-opt", tools);
  const std::filesystem::path importedBase = temporary / "imported";
  const std::filesystem::path imported = temporary / "imported.onnx.mlir";
  const std::filesystem::path preprocessed = temporary / "preprocessed.mlir";
  const std::filesystem::path lowered = temporary / "stablehlo.mlir";
  const std::filesystem::path residual = temporary / "stablehlo.residual.mlir";
  const std::filesystem::path stagedOutput = temporary / "publish.mlir";
  const std::filesystem::path stagedMetadata = temporary / "boundary.json";
  const std::vector<std::string> importArgs{
      "--EmitONNXIR", "--mlir-print-debuginfo", "-o", importedBase.string(),
      model.string()};
  const std::vector<std::string> preprocessArgs{
      "--allow-unregistered-dialect",
      "--mlir-print-op-generic",
      "--mlir-print-debuginfo",
      "--onnx-string-to-framework-custom-call",
      "--onnx-slice-preprocess",
      "--onnx-dynamic-tile-preprocess",
      imported.string(),
      "-o",
      preprocessed.string()};
  const std::vector<std::string> lowerArgs{"--allow-unregistered-dialect",
                                           "--convert-onnx-to-stablehlo",
                                           "--mlir-print-op-generic",
                                           "--mlir-print-debuginfo",
                                           preprocessed.string(),
                                           "-o",
                                           lowered.string()};
  const std::vector<std::string> residualArgs{
      "--allow-unregistered-dialect",
      "--framework-custom-call-to-stablehlo",
      "--onnx-external-to-stablehlo",
      "--framework-shape-to-stablehlo",
      "--canonicalize",
      "--verify-framework-stablehlo",
      "--mlir-print-op-generic",
      "--mlir-print-debuginfo",
      lowered.string(),
      "-o",
      residual.string()};

  const int importCode =
      runStage("ONNX import", importer, importArgs, temporary);
  if (importCode != 0 || !std::filesystem::is_regular_file(imported)) {
    if (importCode == 0)
      llvm::errs() << "annc-onnx2stablehlo: ONNX importer did not create "
                   << imported << "\n";
    cleanup();
    return importCode == 0 ? 1 : importCode;
  }
  const std::string helper =
      executablePath(frameworkOpt, "annc-onnx-framework-opt", tools);
  const bool helperAvailable =
      !frameworkOpt.empty() || std::filesystem::is_regular_file(helper);
  if (!helperAvailable) {
    llvm::errs() << "annc-onnx2stablehlo: framework optimizer is required; use "
                    "--framework-opt\n";
    cleanup();
    return 1;
  }
  int preprocessCode = 0;
  if (helperAvailable) {
    preprocessCode =
        runStage("ONNX string custom-call adaptation and shape preprocessing",
                 helper, preprocessArgs, temporary);
    if (preprocessCode != 0 ||
        !std::filesystem::is_regular_file(preprocessed)) {
      if (preprocessCode == 0)
        llvm::errs() << "annc-onnx2stablehlo: preprocessing did not create "
                     << preprocessed << "\n";
      cleanup();
      return preprocessCode == 0 ? 1 : preprocessCode;
    }
  } else {
    std::error_code ec;
    std::filesystem::copy_file(
        imported, preprocessed,
        std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
      llvm::errs() << "annc-onnx2stablehlo: cannot stage importer output: "
                   << ec.message() << "\n";
      cleanup();
      return 1;
    }
  }
  const int lowerCode =
      runStage("StableHLO lowering", optimizer, lowerArgs, temporary);
  if (lowerCode != 0 || !std::filesystem::is_regular_file(lowered)) {
    if (lowerCode == 0)
      llvm::errs() << "annc-onnx2stablehlo: optimizer did not create "
                   << lowered << "\n";
    cleanup();
    return lowerCode == 0 ? 1 : lowerCode;
  }
  int residualCode = 0;
  if (helperAvailable) {
    residualCode =
        runStage("residual ONNX lowering", helper, residualArgs, temporary);
    if (residualCode != 0 || !std::filesystem::is_regular_file(residual)) {
      if (residualCode == 0)
        llvm::errs() << "annc-onnx2stablehlo: residual lowering did not create "
                     << residual << "\n";
      cleanup();
      return residualCode == 0 ? 1 : residualCode;
    }
  }
  const std::filesystem::path finalInput = helperAvailable ? residual : lowered;

  std::ifstream input(finalInput);
  const std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  const std::string cleaned = stripEntryPoint(text);
  {
    std::ofstream rewritten(stagedOutput, std::ios::trunc);
    rewritten << cleaned;
    rewritten.close();
    if (!rewritten) {
      llvm::errs() << "annc-onnx2stablehlo: cannot write staged output\n";
      cleanup();
      return 1;
    }
  }
  const Counts counts = countText(cleaned);
  const std::string producer = producerVersion.empty()
                                   ? envOrUnknown("ANNC_ONNX_MLIR_VERSION")
                                   : producerVersion.getValue();
  const std::string stablehlo =
      stablehloVersion.empty() ? envOrUnknown("ANNC_STABLEHLO_PRODUCER_VERSION")
                               : stablehloVersion.getValue();
  if (!writeMetadata(stagedMetadata, model, output, imported, finalInput,
                     importer, optimizer, helper, producer, stablehlo,
                     importArgs, preprocessArgs, lowerArgs, residualArgs,
                     importCode, preprocessCode, lowerCode, residualCode,
                     helperAvailable, counts)) {
    llvm::errs() << "annc-onnx2stablehlo: cannot stage boundary artifacts\n";
    cleanup();
    return 1;
  }

  std::error_code renameError;
  const bool hadOutput = std::filesystem::exists(output);
  const std::filesystem::path backup = temporary / "previous-output.mlir";
  if (hadOutput) {
    std::filesystem::copy_file(output, backup, renameError);
    if (renameError) {
      llvm::errs() << "annc-onnx2stablehlo: cannot preserve existing output: "
                   << renameError.message() << "\n";
      cleanup();
      return 1;
    }
  }
  std::filesystem::rename(stagedOutput, output, renameError);
  if (renameError) {
    llvm::errs() << "annc-onnx2stablehlo: cannot publish " << output << ": "
                 << renameError.message() << "\n";
    cleanup();
    return 1;
  }
  std::filesystem::rename(stagedMetadata, metadata, renameError);
  if (renameError) {
    llvm::errs() << "annc-onnx2stablehlo: cannot publish " << metadata << ": "
                 << renameError.message() << "\n";
    std::error_code rollbackError;
    if (hadOutput)
      std::filesystem::rename(backup, output, rollbackError);
    else
      std::filesystem::remove(output, rollbackError);
    if (rollbackError) {
      llvm::errs()
          << "annc-onnx2stablehlo: output rollback failed; preserved backup at "
          << backup << ": " << rollbackError.message() << "\n";
      return 1;
    }
    cleanup();
    return 1;
  }
  std::error_code staleFailureError;
  std::filesystem::remove(metadata.string() + ".failure.json",
                          staleFailureError);
  cleanup();
  return 0;
}
