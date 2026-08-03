#include "FusionMetadata/FusionMetadata.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <system_error>
#include <unordered_set>

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

std::vector<int64_t> intArrayAttr(DictionaryAttr dict, StringRef key);

FusionArg fusionArgFromAttr(DictionaryAttr dict) {
  FusionArg arg;
  if (!dict) return arg;
  arg.role = stringAttr(dict, "role");
  arg.tfName = stringAttr(dict, "tf_name");
  arg.shape = intArrayAttr(dict, "shape");
  arg.rank = intAttr(dict, "rank", -1);
  arg.dtype = stringAttr(dict, "dtype");
  return arg;
}

std::vector<FusionArg> fusionArgArrayAttr(DictionaryAttr dict, StringRef key) {
  std::vector<FusionArg> values;
  if (!dict) return values;
  Attribute raw = dict.get(key);
  auto arr = raw ? dyn_cast<ArrayAttr>(raw) : ArrayAttr();
  if (!arr) return values;
  values.reserve(arr.size());
  for (Attribute attr : arr) {
    if (auto item = dyn_cast<DictionaryAttr>(attr)) {
      values.push_back(fusionArgFromAttr(item));
    }
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

FusionArg fusionArgFromMetadataEntries(
    const std::map<std::string, std::string> &entries) {
  FusionArg arg;
  auto value = [&](const char *key) -> StringRef {
    auto it = entries.find(key);
    return it == entries.end() ? StringRef{} : StringRef(it->second);
  };
  arg.role = parseQuotedString(value("role"));
  arg.tfName = parseQuotedString(value("tf_name"));
  arg.shape = parseIntegerArray(value("shape"));
  if (!parseInteger(value("rank"), &arg.rank)) arg.rank = -1;
  arg.dtype = parseQuotedString(value("dtype"));
  return arg;
}

std::vector<FusionArg> fusionArgArrayFromMetadata(StringRef value) {
  std::vector<FusionArg> values;
  size_t open = value.find('[');
  size_t close = value.rfind(']');
  if (open == StringRef::npos || close == StringRef::npos || close <= open) {
    return values;
  }
  for (const std::string &part :
       splitTopLevel(value.slice(open + 1, close), ',')) {
    StringRef partRef(part);
    size_t dictOpen = partRef.find('{');
    size_t dictClose = partRef.rfind('}');
    if (dictOpen == StringRef::npos || dictClose == StringRef::npos ||
        dictClose <= dictOpen) {
      continue;
    }
    auto entries =
        parseMetadataDictionary(partRef.slice(dictOpen + 1, dictClose));
    if (!entries.empty()) {
      values.push_back(fusionArgFromMetadataEntries(entries));
    }
  }
  return values;
}

FusionInfo fusionInfoFromMetadataEntries(
    const std::map<std::string, std::string> &entries, StringRef attrContext) {
  FusionInfo info;
  auto rawValue = [&](const char *key) -> StringRef {
    auto it = entries.find(key);
    return it == entries.end() ? StringRef{} : StringRef(it->second);
  };
  auto stringValue = [&](const char *key) -> std::string {
    return parseQuotedString(rawValue(key));
  };
  auto intArrayValue = [&](const char *key) -> std::vector<int64_t> {
    return parseIntegerArray(rawValue(key));
  };

  info.name = stringValue("tf.name");
  info.pattern = stringValue("fusion.pattern");
  info.kernelName = stringValue("kernel_name");
  info.args = fusionArgArrayFromMetadata(rawValue("args"));
  info.outputs = fusionArgArrayFromMetadata(rawValue("outputs"));
  info.abi = stringValue("abi");
  if (info.abi.empty()) info.abi = "mlir_ciface";
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

  return info;
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

llvm::Error validateFusionInfo(const FusionInfo &info) {
  if (info.name.empty()) return missingFieldError("name");
  if (info.pattern.empty()) return missingFieldError("pattern");
  if (info.kernelName.empty()) return missingFieldError("kernel_name");
  if (info.args.empty()) return missingFieldError("args");
  if (info.outputs.empty()) return missingFieldError("outputs");

  static const std::unordered_set<std::string> inputRoles = {
      "constant", "fixed", "dynamic"};
  for (const FusionArg &arg : info.args) {
    if (!inputRoles.count(arg.role)) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "fusion input '%s' has invalid role '%s'", arg.tfName.c_str(),
          arg.role.c_str());
    }
    if (arg.tfName.empty()) return missingFieldError("args.tf_name");
    if (arg.rank < 0 || arg.rank != static_cast<int64_t>(arg.shape.size())) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "fusion input '%s' rank does not match its shape", arg.tfName.c_str());
    }
    if (arg.dtype.empty()) return missingFieldError("args.dtype");
  }
  for (const FusionArg &output : info.outputs) {
    if (output.role != "output") {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "fusion output '%s' has invalid role '%s'", output.tfName.c_str(),
          output.role.c_str());
    }
    if (output.tfName.empty()) return missingFieldError("outputs.tf_name");
    if (output.rank < 0 ||
        output.rank != static_cast<int64_t>(output.shape.size())) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "fusion output '%s' rank does not match its shape",
          output.tfName.c_str());
    }
    if (output.dtype.empty()) return missingFieldError("outputs.dtype");
  }
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
  info.args = fusionArgArrayAttr(metadata, "args");
  info.outputs = fusionArgArrayAttr(metadata, "outputs");
  info.abi = stringAttr(metadata, "abi");
  if (info.abi.empty()) info.abi = "mlir_ciface";
  info.dynamicDims = intArrayAttr(metadata, "dynamic_dims");
  info.kernelArgOrder = intArrayAttr(metadata, "kernel_arg_order");
  info.symbolicSignature = stringAttr(metadata, "symbolic_signature");
  info.fallbackFunction = stringAttr(metadata, "fallback_function");
  populatePatternAttrs(info, func);

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
