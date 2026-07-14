#include "FusionMetadata/FusionMetadata.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <system_error>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

using namespace mlir;

namespace annc::fusion {
namespace {

std::string stringAttr(DictionaryAttr dict, StringRef key) {
  if (!dict) return "";
  Attribute raw = dict.get(key);
  if (auto attr = raw ? dyn_cast<StringAttr>(raw) : StringAttr()) {
    return attr.str();
  }
  return "";
}

int64_t intAttr(DictionaryAttr dict, StringRef key,
                int64_t defaultValue = 0) {
  if (!dict) return defaultValue;
  Attribute raw = dict.get(key);
  if (auto attr = raw ? dyn_cast<IntegerAttr>(raw) : IntegerAttr()) {
    return attr.getInt();
  }
  return defaultValue;
}

std::vector<std::string> stringArrayAttr(DictionaryAttr dict, StringRef key) {
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

std::vector<int64_t> intArrayAttr(DictionaryAttr dict, StringRef key) {
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

std::vector<int64_t> parseShapeString(const std::string &shape) {
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

std::vector<std::vector<int64_t>> parseShapeStrings(
    const std::vector<std::string> &shapes) {
  std::vector<std::vector<int64_t>> parsed;
  parsed.reserve(shapes.size());
  for (const auto &shape : shapes) parsed.push_back(parseShapeString(shape));
  return parsed;
}

std::string trim(std::string value) {
  auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
    return std::isspace(c);
  });
  auto last = std::find_if_not(value.rbegin(), value.rend(),
                              [](unsigned char c) { return std::isspace(c); })
                  .base();
  if (first >= last) return "";
  return std::string(first, last);
}

size_t findMatchingBrace(StringRef text, size_t openPos) {
  int depth = 0;
  bool inString = false;
  bool escaped = false;
  for (size_t i = openPos; i < text.size(); ++i) {
    char c = text[i];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        inString = false;
      }
      continue;
    }
    if (c == '"') {
      inString = true;
    } else if (c == '{') {
      ++depth;
    } else if (c == '}') {
      --depth;
      if (depth == 0) return i;
    }
  }
  return StringRef::npos;
}

std::vector<std::string> splitTopLevel(StringRef text, char delimiter) {
  std::vector<std::string> parts;
  size_t start = 0;
  int braceDepth = 0;
  int bracketDepth = 0;
  bool inString = false;
  bool escaped = false;
  for (size_t i = 0; i < text.size(); ++i) {
    char c = text[i];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        inString = false;
      }
      continue;
    }
    if (c == '"') {
      inString = true;
    } else if (c == '{') {
      ++braceDepth;
    } else if (c == '}') {
      --braceDepth;
    } else if (c == '[') {
      ++bracketDepth;
    } else if (c == ']') {
      --bracketDepth;
    } else if (c == delimiter && braceDepth == 0 && bracketDepth == 0) {
      parts.push_back(trim(text.slice(start, i).str()));
      start = i + 1;
    }
  }
  parts.push_back(trim(text.drop_front(start).str()));
  return parts;
}

std::string parseQuotedString(StringRef value) {
  size_t begin = value.find('"');
  if (begin == StringRef::npos) return "";
  std::string result;
  bool escaped = false;
  for (size_t i = begin + 1; i < value.size(); ++i) {
    char c = value[i];
    if (escaped) {
      result.push_back(c);
      escaped = false;
    } else if (c == '\\') {
      escaped = true;
    } else if (c == '"') {
      return result;
    } else {
      result.push_back(c);
    }
  }
  return "";
}

std::vector<std::string> parseQuotedStringArray(StringRef value) {
  std::vector<std::string> result;
  size_t open = value.find('[');
  size_t close = value.rfind(']');
  if (open == StringRef::npos || close == StringRef::npos || close <= open) {
    return result;
  }
  for (const std::string &part :
       splitTopLevel(value.slice(open + 1, close), ',')) {
    std::string parsed = parseQuotedString(part);
    if (!parsed.empty()) result.push_back(parsed);
  }
  return result;
}

bool parseInteger(StringRef value, int64_t *out) {
  size_t pos = 0;
  while (pos < value.size() &&
         !(std::isdigit(static_cast<unsigned char>(value[pos])) ||
           value[pos] == '-')) {
    ++pos;
  }
  if (pos == value.size()) return false;
  size_t end = pos + 1;
  while (end < value.size() &&
         std::isdigit(static_cast<unsigned char>(value[end]))) {
    ++end;
  }
  *out = std::stoll(value.slice(pos, end).str());
  return true;
}

std::vector<int64_t> parseIntegerArray(StringRef value) {
  std::vector<int64_t> result;
  size_t open = value.find('[');
  size_t close = value.rfind(']');
  if (open == StringRef::npos || close == StringRef::npos || close <= open) {
    return result;
  }
  for (const std::string &part :
       splitTopLevel(value.slice(open + 1, close), ',')) {
    int64_t parsed = 0;
    if (parseInteger(part, &parsed)) result.push_back(parsed);
  }
  return result;
}

std::map<std::string, std::string> parseMetadataDictionary(StringRef body) {
  std::map<std::string, std::string> entries;
  for (const std::string &entry : splitTopLevel(body, ',')) {
    size_t eq = entry.find('=');
    if (eq == std::string::npos) continue;
    entries[trim(entry.substr(0, eq))] = trim(entry.substr(eq + 1));
  }
  return entries;
}

FusionInfo fusionInfoFromMetadataEntries(
    const std::map<std::string, std::string> &entries, StringRef attrContext) {
  FusionInfo info;
  auto stringValue = [&](const char *key) -> std::string {
    auto it = entries.find(key);
    return it == entries.end() ? "" : parseQuotedString(it->second);
  };
  auto stringArrayValue = [&](const char *key) -> std::vector<std::string> {
    auto it = entries.find(key);
    return it == entries.end() ? std::vector<std::string>{}
                               : parseQuotedStringArray(it->second);
  };
  auto intValue = [&](const char *key, int64_t fallback = 0) -> int64_t {
    auto it = entries.find(key);
    int64_t parsed = fallback;
    if (it != entries.end()) parseInteger(it->second, &parsed);
    return parsed;
  };
  auto intArrayValue = [&](const char *key) -> std::vector<int64_t> {
    auto it = entries.find(key);
    return it == entries.end() ? std::vector<int64_t>{}
                               : parseIntegerArray(it->second);
  };

  info.name = stringValue("tf.name");
  info.pattern = stringValue("fusion.pattern");
  info.kernelName = stringValue("kernel_name");
  info.outputTensor = stringValue("tf.output");
  info.originalNodes = stringArrayValue("tf.nodes");
  info.inputs = stringArrayValue("tf.inputs");
  info.inputShapes = parseShapeStrings(stringArrayValue("tf.input_shapes"));
  info.outputShape = parseShapeString(stringValue("tf.output_shape"));
  info.abi = stringValue("abi");
  if (info.abi.empty()) info.abi = "mlir_ciface";
  info.nConstants = intValue("Nconstants");
  info.nFixed = intValue("Nfixed");
  info.nDynamic = intValue("Ndynamic");
  info.numOutputs = intValue("num_outputs", 1);
  info.outputRanks = intArrayValue("output_ranks");
  info.inputRanks = intArrayValue("input_ranks");
  info.dynamicDims = intArrayValue("dynamic_dims");
  info.kernelArgOrder = intArrayValue("kernel_arg_order");
  info.symbolicSignature = stringValue("symbolic_signature");
  info.fallbackFunction = stringValue("fallback_function");

  size_t numBuckets = attrContext.find("fusion.num_buckets");
  if (numBuckets != StringRef::npos) {
    size_t eq = attrContext.find('=', numBuckets);
    if (eq != StringRef::npos) {
      int64_t parsed = 0;
      if (parseInteger(attrContext.drop_front(eq + 1), &parsed)) {
        info.patternAttrs["num_buckets"] = std::to_string(parsed);
      }
    }
  }

  normalizeFusionInfo(info);
  return info;
}

int64_t rankOf(ArrayRef<int64_t> shape) {
  return static_cast<int64_t>(shape.size());
}

FusionArg makeFusionArg(StringRef role, StringRef tfName,
                        ArrayRef<int64_t> shape) {
  FusionArg arg;
  arg.role = role.str();
  arg.tfName = tfName.str();
  arg.shape.assign(shape.begin(), shape.end());
  arg.rank = rankOf(shape);
  return arg;
}

std::string roleForInputIndex(const FusionInfo &info, size_t index) {
  if (index < static_cast<size_t>(std::max<int64_t>(info.nConstants, 0))) {
    return "constant";
  }
  index -= static_cast<size_t>(std::max<int64_t>(info.nConstants, 0));
  if (index < static_cast<size_t>(std::max<int64_t>(info.nFixed, 0))) {
    return "fixed";
  }
  index -= static_cast<size_t>(std::max<int64_t>(info.nFixed, 0));
  if (index < static_cast<size_t>(std::max<int64_t>(info.nDynamic, 0))) {
    return "dynamic";
  }
  return "input";
}

void populateContractFromLegacy(FusionInfo &info) {
  if (info.args.empty()) {
    info.args.reserve(info.inputs.size());
    for (size_t i = 0; i < info.inputs.size(); ++i) {
      ArrayRef<int64_t> shape;
      if (i < info.inputShapes.size()) shape = info.inputShapes[i];
      info.args.push_back(makeFusionArg(roleForInputIndex(info, i),
                                        info.inputs[i], shape));
    }
  }

  if (info.outputs.empty() && !info.outputTensor.empty()) {
    info.outputs.push_back(
        makeFusionArg("output", info.outputTensor, info.outputShape));
  }
}

void populateLegacyFromContract(FusionInfo &info) {
  if (info.inputs.empty()) {
    for (const FusionArg &arg : info.args) info.inputs.push_back(arg.tfName);
  }
  if (info.inputShapes.empty()) {
    for (const FusionArg &arg : info.args) info.inputShapes.push_back(arg.shape);
  }
  if (info.inputRanks.empty()) {
    for (const FusionArg &arg : info.args) {
      info.inputRanks.push_back(arg.rank >= 0 ? arg.rank : rankOf(arg.shape));
    }
  }

  if (info.outputShape.empty() && !info.outputs.empty()) {
    info.outputShape = info.outputs.front().shape;
  }
  if (info.outputRanks.empty() && !info.outputs.empty()) {
    for (const FusionArg &output : info.outputs) {
      info.outputRanks.push_back(output.rank >= 0 ? output.rank
                                                  : rankOf(output.shape));
    }
  }
  if (info.outputTensor.empty() && !info.outputs.empty()) {
    info.outputTensor = info.outputs.front().tfName;
  }

  if (info.nConstants == 0 && info.nFixed == 0 && info.nDynamic == 0) {
    for (const FusionArg &arg : info.args) {
      if (arg.role == "constant") {
        ++info.nConstants;
      } else if (arg.role == "fixed") {
        ++info.nFixed;
      } else if (arg.role == "dynamic") {
        ++info.nDynamic;
      }
    }
  }
  if (info.numOutputs == 1 && !info.outputs.empty()) {
    info.numOutputs = static_cast<int64_t>(info.outputs.size());
  }
}

void populatePatternAttrs(FusionInfo &info, func::FuncOp func) {
  if (auto numBuckets = func->getAttrOfType<IntegerAttr>("fusion.num_buckets")) {
    info.patternAttrs["num_buckets"] = std::to_string(numBuckets.getInt());
  }
}

llvm::Error missingFieldError(StringRef field) {
  return llvm::createStringError(std::errc::invalid_argument,
                                 "fusion metadata missing required field '%s'",
                                 field.str().c_str());
}

}  // namespace

void normalizeFusionInfo(FusionInfo &info) {
  populateLegacyFromContract(info);
  populateContractFromLegacy(info);
}

llvm::Error validateFusionInfo(const FusionInfo &info) {
  if (info.name.empty()) return missingFieldError("name");
  if (info.inputs.empty() && info.args.empty()) {
    return missingFieldError("inputs");
  }
  if (info.outputTensor.empty()) return missingFieldError("output_tensor");
  return llvm::Error::success();
}

llvm::Expected<FusionInfo> readFusionInfo(func::FuncOp func) {
  if (!func->hasAttr("annc.kernel")) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "function is not an ANNC kernel");
  }

  auto metadata = func->getAttrOfType<DictionaryAttr>("fusion.metadata");
  if (!metadata) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "function has no fusion.metadata");
  }

  FusionInfo info;
  info.name = stringAttr(metadata, "tf.name");
  info.pattern = stringAttr(metadata, "fusion.pattern");
  info.kernelName = stringAttr(metadata, "kernel_name");
  info.outputTensor = stringAttr(metadata, "tf.output");
  info.originalNodes = stringArrayAttr(metadata, "tf.nodes");
  info.inputs = stringArrayAttr(metadata, "tf.inputs");
  info.inputShapes =
      parseShapeStrings(stringArrayAttr(metadata, "tf.input_shapes"));
  info.outputShape = parseShapeString(stringAttr(metadata, "tf.output_shape"));
  info.abi = stringAttr(metadata, "abi");
  if (info.abi.empty()) info.abi = "mlir_ciface";
  info.nConstants = intAttr(metadata, "Nconstants");
  info.nFixed = intAttr(metadata, "Nfixed");
  info.nDynamic = intAttr(metadata, "Ndynamic");
  info.numOutputs = intAttr(metadata, "num_outputs", 1);
  info.outputRanks = intArrayAttr(metadata, "output_ranks");
  info.inputRanks = intArrayAttr(metadata, "input_ranks");
  info.dynamicDims = intArrayAttr(metadata, "dynamic_dims");
  info.kernelArgOrder = intArrayAttr(metadata, "kernel_arg_order");
  info.symbolicSignature = stringAttr(metadata, "symbolic_signature");
  info.fallbackFunction = stringAttr(metadata, "fallback_function");
  populatePatternAttrs(info, func);
  normalizeFusionInfo(info);

  if (auto err = validateFusionInfo(info)) return std::move(err);
  return info;
}

llvm::Expected<std::vector<FusionInfo>> extractFusionInfos(ModuleOp module) {
  std::vector<FusionInfo> infos;
  llvm::Error errors = llvm::Error::success();

  module.walk([&](func::FuncOp funcOp) {
    if (!funcOp->hasAttr("annc.kernel")) return;
    auto info = readFusionInfo(funcOp);
    if (!info) {
      errors = llvm::joinErrors(std::move(errors), info.takeError());
      return;
    }
    infos.push_back(std::move(*info));
  });

  if (errors) return std::move(errors);
  if (infos.empty()) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "no ANNC fusion metadata found");
  }
  return infos;
}

llvm::Expected<std::vector<FusionInfo>> extractFusionInfosFromMlirText(
    const std::string &path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    return llvm::createStringError(std::errc::io_error,
                                   "cannot open ATIR file '%s'",
                                   path.c_str());
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  std::string storage = buffer.str();
  StringRef text(storage);

  std::vector<FusionInfo> infos;
  size_t searchPos = 0;
  while (true) {
    size_t metadataPos = text.find("fusion.metadata", searchPos);
    if (metadataPos == StringRef::npos) break;
    size_t eq = text.find('=', metadataPos);
    size_t open = eq == StringRef::npos ? StringRef::npos : text.find('{', eq);
    if (open == StringRef::npos) {
      return llvm::createStringError(std::errc::invalid_argument,
                                     "malformed fusion.metadata in '%s'",
                                     path.c_str());
    }
    size_t close = findMatchingBrace(text, open);
    if (close == StringRef::npos) {
      return llvm::createStringError(std::errc::invalid_argument,
                                     "unterminated fusion.metadata in '%s'",
                                     path.c_str());
    }

    size_t attrStart = text.take_front(metadataPos).rfind("attributes");
    if (attrStart == StringRef::npos) attrStart = metadataPos;
    StringRef attrContext = text.slice(attrStart, close + 1);
    auto entries = parseMetadataDictionary(text.slice(open + 1, close));
    FusionInfo info = fusionInfoFromMetadataEntries(entries, attrContext);
    if (auto err = validateFusionInfo(info)) return std::move(err);
    infos.push_back(std::move(info));
    searchPos = close + 1;
  }

  if (infos.empty()) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "no ANNC fusion metadata found in '%s'",
                                   path.c_str());
  }
  return infos;
}

}  // namespace annc::fusion
