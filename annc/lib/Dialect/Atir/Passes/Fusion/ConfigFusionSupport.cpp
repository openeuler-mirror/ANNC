#include "ConfigFusionSupport.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <unordered_set>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/SHA256.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "nlohmann/json.hpp"

using namespace llvm;
using namespace mlir;
using namespace atir;

namespace atir::config_fusion {

// Gate debug logging with an environment variable so normal pass output stays
// readable.
bool configFusionDebugEnabled() {
  static bool enabled = [] {
    const char *value = std::getenv("ANNC_CONFIG_FUSION_DEBUG");
    return value && std::strcmp(value, "0") != 0;
  }();
  return enabled;
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

std::vector<std::string> splitTopLevel(StringRef text, char delimiter) {
  std::vector<std::string> parts;
  size_t start = 0;
  int parenDepth = 0;
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
    } else if (c == '(') {
      ++parenDepth;
    } else if (c == ')') {
      --parenDepth;
    } else if (c == '[') {
      ++bracketDepth;
    } else if (c == ']') {
      --bracketDepth;
    } else if (c == delimiter && parenDepth == 0 && bracketDepth == 0) {
      parts.push_back(trim(text.slice(start, i).str()));
      start = i + 1;
    }
  }
  parts.push_back(trim(text.drop_front(start).str()));
  return parts;
}

// The legacy capture DSL uses attr(alias, "attrName"); extract the first
// quoted string here. The structured capture schema bypasses this path.
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

// nlohmann/json rejects trailing commas by default. Configs are hand-written,
// so remove commas before closing objects/arrays without touching strings.
static std::string stripJsonTrailingCommas(StringRef text) {
  std::string out;
  out.reserve(text.size());
  bool inString = false;
  bool escaped = false;
  for (size_t i = 0; i < text.size(); ++i) {
    char c = text[i];
    if (inString) {
      out.push_back(c);
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
      out.push_back(c);
      continue;
    }
    if (c == ',') {
      size_t next = i + 1;
      while (next < text.size() &&
             std::isspace(static_cast<unsigned char>(text[next]))) {
        ++next;
      }
      if (next < text.size() && (text[next] == ']' || text[next] == '}')) {
        continue;
      }
    }
    out.push_back(c);
  }
  return out;
}

bool parseInteger(StringRef value, int64_t *out) {
  size_t pos = 0;
  while (pos < value.size() &&
         !(std::isdigit(static_cast<unsigned char>(value[pos])) ||
           value[pos] == '-')) {
    ++pos;
  }
  if (pos == value.size()) return false;
  size_t digitsBegin = pos;
  if (value[pos] == '-') ++digitsBegin;
  if (digitsBegin == value.size() ||
      !std::isdigit(static_cast<unsigned char>(value[digitsBegin]))) {
    return false;
  }
  size_t end = digitsBegin + 1;
  while (end < value.size() &&
         std::isdigit(static_cast<unsigned char>(value[end]))) {
    ++end;
  }
  return !value.slice(pos, end).getAsInteger(10, *out);
}

// MLIR op names are usually shaped like atir.MatMul; the DSL writes MatMul.
std::string simpleOpName(Operation *op) {
  if (!op) return "";
  StringRef full = op->getName().getStringRef();
  size_t dot = full.rfind('.');
  return dot == StringRef::npos ? full.str()
                                : full.drop_front(dot + 1).str();
}

// Op names in config files are case-insensitive. Known ATIR ops are normalized
// to canonical simple names; unknown names are preserved for extension/debugging.
std::string canonicalConfigOpName(StringRef opName) {
  std::string trimmed = trim(opName.str());
  std::string lower = StringRef(trimmed).lower();
  return llvm::StringSwitch<llvm::StringRef>(lower)
      .Case("abs", "Abs")
      .Case("add", "Add")
      .Case("batchmatmul", "BatchMatMul")
      .Case("broadcast", "broadcast")
      .Case("cast", "Cast")
      .Case("compare", "Compare")
      .Case("concat", "Concat")
      .Case("concatv2", "ConcatV2")
      .Case("dot", "Dot")
      .Case("expanddims", "ExpandDims")
      .Case("fill", "Fill")
      .Case("gather", "Gather")
      .Case("gathernd", "GatherNd")
      .Case("identity", "Identity")
      .Case("logistic", "Logistic")
      .Case("matmul", "MatMul")
      .Case("maximum", "Maximum")
      .Case("minimum", "Minimum")
      .Case("mul", "Mul")
      .Case("pack", "Pack")
      .Case("pad", "Pad")
      .Case("pow", "Pow")
      .Case("prod", "Prod")
      .Case("realdiv", "RealDiv")
      .Case("reducemean", "ReduceMean")
      .Case("relu", "Relu")
      .Case("reshape", "Reshape")
      .Case("resourcegather", "ResourceGather")
      .Case("rsqrt", "Rsqrt")
      .Case("slice", "Slice")
      .Case("softmax", "Softmax")
      .Case("sparsesegmentmean", "SparseSegmentMean")
      .Case("sparsesegmentmin", "SparseSegmentMin")
      .Case("sparsesegmentsum", "SparseSegmentSum")
      .Case("sparsefillemptyrows", "SparseFillEmptyRows")
      .Case("sparsereshape", "SparseReshape")
      .Case("sparsetodense", "SparseToDense")
      .Case("stridedslice", "StridedSlice")
      .Case("stringtohashbucketfast", "StringToHashBucketFast")
      .Case("sub", "Sub")
      .Case("sum", "Sum")
      .Case("tensorscatterupdate", "TensorScatterUpdate")
      .Case("tile", "Tile")
      .Case("topk", "TopK")
      .Case("transpose", "Transpose")
      .Case("unique", "Unique")
      .Case("unsortedsegmentmin", "UnsortedSegmentMin")
      .Case("where", "Where")
      .Case("zeroslike", "ZerosLike")
      .Default(trimmed)
      .str();
}

std::string sanitizeName(std::string name) {
  for (char &c : name) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
  }
  return name;
}

// Allocate a symbol name for a generated kernel func that is unique in the module.
std::string uniquifySymbolName(ModuleOp module, StringRef baseName) {
  std::string unique = baseName.str();
  unsigned suffix = 1;
  while (module.lookupSymbol(unique)) {
    unique = (Twine(baseName) + "_" + Twine(suffix++)).str();
  }
  return unique;
}

// Recover a TensorFlow-side tensor name when possible: prefer the ATIR tensor
// type name, then a block-argument location, and finally the defining op name.
std::string getValueName(Value value) {
  if (auto tensorType = dyn_cast<atir::TensorType>(value.getType())) {
    std::string name = tensorType.getValueOfName();
    if (!name.empty()) return name;
  }
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (auto loc = dyn_cast<NameLoc>(blockArg.getLoc())) {
      return loc.getName().str();
    }
    return (Twine("arg") + Twine(blockArg.getArgNumber())).str();
  }
  if (Operation *op = value.getDefiningOp()) {
    if (auto dict = op->getAttrOfType<DictionaryAttr>("metadata")) {
      if (auto attr = dyn_cast_or_null<StringAttr>(dict.get("tf.name"))) {
        return attr.str();
      }
    }
    return simpleOpName(op);
  }
  return "";
}

std::string getFusionOutputName(Value value) {
  std::string name;
  if (auto result = dyn_cast<OpResult>(value)) {
    if (auto metadata =
            result.getOwner()->getAttrOfType<DictionaryAttr>("metadata")) {
      if (auto endpoints =
              dyn_cast_or_null<ArrayAttr>(metadata.get("tf.output_tensors"));
          endpoints && result.getResultNumber() < endpoints.size()) {
        if (auto endpoint =
                dyn_cast<StringAttr>(endpoints[result.getResultNumber()]);
            endpoint && !endpoint.getValue().empty()) {
          name = endpoint.str();
        }
      }
    }
    if (name.empty()) name = getValueName(value);
    if (!name.empty() && name.find(':') == std::string::npos) {
      name += ":" + std::to_string(result.getResultNumber());
    }
    return name;
  }
  return getValueName(value);
}

// Convert ATIR tensor/scalar types to the dtype text used by fusion.metadata.
std::string getTensorDType(Type type) {
  if (auto tensorType = dyn_cast<atir::TensorType>(type)) {
    if (auto encoding = dyn_cast_or_null<StringAttr>(tensorType.getEncoding())) {
      if (!encoding.getValue().empty()) return encoding.str();
    }
    type = tensorType.getElementType();
  }
  if (auto floatType = dyn_cast<FloatType>(type)) {
    if (floatType.isF16()) return "f16";
    if (floatType.isF32()) return "f32";
    if (floatType.isF64()) return "f64";
  }
  if (auto intType = dyn_cast<IntegerType>(type)) {
    StringRef prefix =
        intType.isSigned() ? "si" : intType.isUnsigned() ? "ui" : "i";
    return (Twine(prefix) + Twine(intType.getWidth())).str();
  }
  if (type.isIndex()) return "index";
  return "";
}

int64_t getRank(Type type) {
  if (auto tensorType = dyn_cast<atir::TensorType>(type)) {
    return static_cast<int64_t>(tensorType.getShape().size());
  }
  return 0;
}

std::vector<int64_t> getShape(Type type) {
  if (auto tensorType = dyn_cast<atir::TensorType>(type)) {
    return std::vector<int64_t>(tensorType.getShape().begin(),
                                tensorType.getShape().end());
  }
  return {};
}

static SmallVector<std::string> parseAliasList(StringRef text) {
  SmallVector<std::string> aliases;
  for (const std::string &part : splitTopLevel(text, ',')) {
    std::string alias = trim(part);
    if (!alias.empty()) aliases.push_back(alias);
  }
  return aliases;
}

// Parse one graph DSL line. The parser handles syntax, whitespace, and trailing
// commas; the matcher validates aliases and operation compatibility.
static bool parseGraphLine(StringRef line, PatternStmt *stmt) {
  std::string text = trim(line.str());
  if (text.empty()) return false;
  size_t eq = text.find('=');
  if (eq == std::string::npos) return false;
  std::string lhs = trim(text.substr(0, eq));
  std::string rhs = trim(text.substr(eq + 1));
  if (lhs.empty() || rhs.empty()) return false;

  if (lhs.front() == '(') {
    size_t close = lhs.rfind(')');
    if (close == std::string::npos || close <= 1) return false;
    if (!trim(lhs.substr(close + 1)).empty()) return false;
    stmt->aliases = parseAliasList(StringRef(lhs).slice(1, close));
  } else {
    stmt->aliases.push_back(trim(lhs));
  }

  size_t paren = rhs.find('(');
  std::string opName =
      trim(paren == std::string::npos ? rhs : rhs.substr(0, paren));
  if (opName.empty()) return false;
  stmt->opName = canonicalConfigOpName(opName);
  if (paren != std::string::npos) {
    size_t close = rhs.rfind(')');
    if (close == std::string::npos || close < paren) return false;
    if (!trim(rhs.substr(close + 1)).empty()) return false;
    std::string args = rhs.substr(paren + 1, close - paren - 1);
    for (const std::string &part : splitTopLevel(args, ',')) {
      std::string arg = trim(part);
      if (!arg.empty()) stmt->args.push_back(arg);
    }
  }
  return true;
}

// Support the legacy capture syntax:
//   "num_buckets = attr(lookup, \"numBuckets\")"
bool parseCaptureEntry(StringRef entry, std::string *key,
                       std::pair<std::string, std::string> *spec) {
  size_t eq = entry.find('=');
  if (eq == StringRef::npos) return false;
  *key = trim(entry.slice(0, eq).str());
  std::string rhs = trim(entry.drop_front(eq + 1).str());
  size_t open = rhs.find("attr(");
  size_t close = rhs.rfind(')');
  if (open == std::string::npos || close == std::string::npos ||
      close <= open) {
    return false;
  }
  std::string inner = rhs.substr(open + 5, close - open - 5);
  std::vector<std::string> parts = splitTopLevel(inner, ',');
  if (parts.size() != 2) return false;
  spec->first = trim(parts[0]);
  spec->second = parseQuotedString(parts[1]);
  return !key->empty() && !spec->first.empty() && !spec->second.empty();
}

// attrs accepts strings, numbers, and booleans; metadata stores them as strings.
static std::string jsonScalarToString(const nlohmann::json &value) {
  if (value.is_string()) return value.get<std::string>();
  if (value.is_boolean()) return value.get<bool>() ? "true" : "false";
  if (value.is_number_integer()) return std::to_string(value.get<int64_t>());
  if (value.is_number_unsigned()) return std::to_string(value.get<uint64_t>());
  if (value.is_number_float()) {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << value.get<double>();
    return os.str();
  }
  return "";
}

static llvm::Expected<std::optional<std::string>> getOptionalStringField(
    const nlohmann::json &object, StringRef field, StringRef context) {
  auto it = object.find(field.str());
  if (it == object.end()) return std::nullopt;
  if (!it->is_string()) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s.%s must be a string",
                                   context.str().c_str(), field.str().c_str());
  }
  return it->get<std::string>();
}

static llvm::Expected<int64_t> getOptionalI64Field(
    const nlohmann::json &object, StringRef field, StringRef context,
    int64_t defaultValue) {
  auto it = object.find(field.str());
  if (it == object.end()) return defaultValue;
  if (it->is_number_integer()) return it->get<int64_t>();
  if (it->is_number_unsigned()) {
    uint64_t value = it->get<uint64_t>();
    if (value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return static_cast<int64_t>(value);
    }
  }
  return llvm::createStringError(std::errc::invalid_argument,
                                 "%s.%s must be an int64",
                                 context.str().c_str(), field.str().c_str());
}

// Convert a legacy string capture into a structured CaptureSpec.
static llvm::Expected<CaptureSpec> parseCaptureString(const std::string &text,
                                                      StringRef ruleName) {
  std::string key;
  std::pair<std::string, std::string> spec;
  if (!parseCaptureEntry(text, &key, &spec)) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "pattern '%s' has malformed capture entry: %s",
        ruleName.str().c_str(), text.c_str());
  }
  return CaptureSpec{key, spec.first, spec.second};
}

// Recommended structured form:
//   "captures": { "key": { "alias": "op_alias", "attr": "attrName" } }
// Array form is also accepted, with "op" allowed as an alias for "alias".
static llvm::Expected<CaptureSpec> parseCaptureObject(
    const nlohmann::json &value, StringRef ruleName,
    StringRef fallbackKey = {}) {
  if (!value.is_object()) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "pattern '%s' capture entry is not an object",
                                   ruleName.str().c_str());
  }

  CaptureSpec capture;
  std::string context =
      (Twine("pattern '") + ruleName + "' capture object").str();
  auto name = getOptionalStringField(value, "name", context);
  if (!name) return name.takeError();
  auto key = getOptionalStringField(value, "key", context);
  if (!key) return key.takeError();
  auto alias = getOptionalStringField(value, "alias", context);
  if (!alias) return alias.takeError();
  auto attr = getOptionalStringField(value, "attr", context);
  if (!attr) return attr.takeError();
  auto op = getOptionalStringField(value, "op", context);
  if (!op) return op.takeError();

  capture.key = trim(name->value_or(key->value_or(fallbackKey.str())));
  capture.alias = trim(alias->value_or(""));
  capture.attr = trim(attr->value_or(""));
  if (capture.alias.empty()) capture.alias = trim(op->value_or(""));
  if (capture.key.empty() || capture.alias.empty() || capture.attr.empty()) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "pattern '%s' capture object requires name/key, alias, and attr",
        ruleName.str().c_str());
  }
  return capture;
}

// attrs must be an object so arrays/null values are not silently dumped into
// hard-to-consume text.
static llvm::Error parseAttrsObject(const nlohmann::json &attrs,
                                    PatternRule *rule) {
  if (!attrs.is_object()) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "pattern '%s' attrs must be an object",
                                   rule->name.c_str());
  }
  for (auto it = attrs.begin(); it != attrs.end(); ++it) {
    std::string value = jsonScalarToString(it.value());
    if (value.empty() && !it.value().is_string()) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "pattern '%s' attrs.%s must be a string, number, or bool",
          rule->name.c_str(), it.key().c_str());
    }
    rule->attrs[it.key()] = value;
  }
  return llvm::Error::success();
}

// captures supports both legacy string arrays and the new object schema. The
// parser performs the structural conversion so materialization need not parse
// JSON/DSL details.
static llvm::Error parseCaptures(const nlohmann::json &captures,
                                 PatternRule *rule) {
  auto append = [&](llvm::Expected<CaptureSpec> capture) -> llvm::Error {
    if (!capture) return capture.takeError();
    rule->captures.push_back(std::move(*capture));
    return llvm::Error::success();
  };

  if (captures.is_array()) {
    for (const auto &v : captures) {
      if (v.is_string()) {
        if (auto err = append(parseCaptureString(v.get<std::string>(),
                                                 rule->name))) {
          return std::move(err);
        }
        continue;
      }
      if (auto err = append(parseCaptureObject(v, rule->name))) {
        return std::move(err);
      }
    }
    return llvm::Error::success();
  }

  if (captures.is_object()) {
    for (auto it = captures.begin(); it != captures.end(); ++it) {
      if (it.value().is_string()) {
        std::string text =
            (Twine(it.key()) + " = " + it.value().get<std::string>()).str();
        if (auto err = append(parseCaptureString(text, rule->name))) {
          return std::move(err);
        }
        continue;
      }
      if (auto err = append(parseCaptureObject(it.value(), rule->name,
                                               it.key()))) {
        return std::move(err);
      }
    }
    return llvm::Error::success();
  }

  return llvm::createStringError(
      std::errc::invalid_argument,
      "pattern '%s' captures must be an array or object", rule->name.c_str());
}

// Read the complete config: preprocess relaxed JSON, parse the pattern list,
// and sort by priority/specificity. Expected lets the pass forward diagnostics.
llvm::Expected<FusionConfig> parseConfigFile(const std::string &path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    return llvm::createStringError(std::errc::io_error,
                                   "cannot open fusion config '%s'",
                                   path.c_str());
  }

  nlohmann::json json;
  try {
    std::stringstream buffer;
    buffer << in.rdbuf();
    json = nlohmann::json::parse(stripJsonTrailingCommas(buffer.str()));
  } catch (const std::exception &e) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "failed to parse fusion config '%s': %s",
                                   path.c_str(), e.what());
  }

  if (!json.is_object()) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "fusion config '%s' is not an object",
                                   path.c_str());
  }

  try {
    FusionConfig config;
    auto version = getOptionalI64Field(json, "version", "fusion config", 0);
    if (!version) return version.takeError();
    config.version = *version;
    if (config.version != 1) {
      return llvm::createStringError(std::errc::invalid_argument,
                                     "unsupported fusion config version '%d'",
                                     static_cast<int>(config.version));
    }
    if (!json.contains("patterns") || !json["patterns"].is_array()) {
      return llvm::createStringError(std::errc::invalid_argument,
                                     "fusion config missing patterns array");
    }

    for (const auto &item : json["patterns"]) {
      if (!item.is_object()) {
        return llvm::createStringError(std::errc::invalid_argument,
                                       "pattern entry is not an object");
      }

      PatternRule rule;
      auto name = getOptionalStringField(item, "name", "pattern");
      if (!name) return name.takeError();
      rule.name = trim(name->value_or(""));
      if (rule.name.empty()) {
        return llvm::createStringError(std::errc::invalid_argument,
                                       "pattern missing name");
      }
      std::string context = (Twine("pattern '") + rule.name + "'").str();
      auto pattern = getOptionalStringField(item, "pattern", context);
      if (!pattern) return pattern.takeError();
      auto kernel = getOptionalStringField(item, "kernel", context);
      if (!kernel) return kernel.takeError();
      auto abi = getOptionalStringField(item, "abi", context);
      if (!abi) return abi.takeError();
      auto priority = getOptionalI64Field(item, "priority", context, 0);
      if (!priority) return priority.takeError();
      rule.pattern = pattern->value_or(rule.name);
      rule.kernel = kernel->value_or(std::string("fused_") + rule.name);
      rule.abi = abi->value_or("mlir_ciface");
      if (rule.abi != "mlir_ciface" && rule.abi != "annc_execution_v2") {
        return llvm::createStringError(
            std::errc::invalid_argument,
            "%s.abi must be 'mlir_ciface' or 'annc_execution_v2'",
            context.c_str());
      }
      rule.priority = *priority;
      if (!item.contains("graph") || !item["graph"].is_array()) {
        return llvm::createStringError(std::errc::invalid_argument,
                                       "pattern '%s' missing graph array",
                                       rule.name.c_str());
      }
      for (const auto &line : item["graph"]) {
        if (!line.is_string()) {
          return llvm::createStringError(std::errc::invalid_argument,
                                         "pattern '%s' graph entry is not a string",
                                         rule.name.c_str());
        }
        PatternStmt stmt;
        if (!parseGraphLine(line.get_ref<const std::string &>(), &stmt)) {
          return llvm::createStringError(std::errc::invalid_argument,
                                         "pattern '%s' has malformed graph line: %s",
                                         rule.name.c_str(),
                                         line.get_ref<const std::string &>().c_str());
        }
        if (stmt.aliases.empty()) {
          return llvm::createStringError(std::errc::invalid_argument,
                                         "pattern '%s' has empty alias list",
                                         rule.name.c_str());
        }
        for (const std::string &alias : stmt.aliases) {
          if (alias == "_") continue;
          if (rule.aliasToStmt.find(alias) != rule.aliasToStmt.end()) {
            return llvm::createStringError(
                std::errc::invalid_argument,
                "pattern '%s' defines alias '%s' more than once",
                rule.name.c_str(), alias.c_str());
          }
          rule.aliasToStmt[alias] = static_cast<unsigned>(rule.stmts.size());
        }
        rule.specificity += static_cast<int64_t>(stmt.aliases.size() * 2 +
                                                stmt.args.size());
        rule.stmts.push_back(std::move(stmt));
      }
      if (item.contains("inputs")) {
        if (!item["inputs"].is_array()) {
          return llvm::createStringError(std::errc::invalid_argument,
                                         "pattern '%s' inputs must be an array",
                                         rule.name.c_str());
        }
        for (const auto &v : item["inputs"]) {
          if (!v.is_string()) {
            return llvm::createStringError(std::errc::invalid_argument,
                                           "pattern '%s' inputs entry is not a string",
                                           rule.name.c_str());
          }
          rule.inputs.push_back(trim(v.get<std::string>()));
        }
      }
      if (item.contains("outputs")) {
        if (!item["outputs"].is_array()) {
          return llvm::createStringError(std::errc::invalid_argument,
                                         "pattern '%s' outputs must be an array",
                                         rule.name.c_str());
        }
        for (const auto &v : item["outputs"]) {
          if (!v.is_string()) {
            return llvm::createStringError(std::errc::invalid_argument,
                                           "pattern '%s' outputs entry is not a string",
                                           rule.name.c_str());
          }
          rule.outputs.push_back(trim(v.get<std::string>()));
        }
      }
      if (item.contains("attrs")) {
        if (auto err = parseAttrsObject(item["attrs"], &rule)) {
          return std::move(err);
        }
      }
      if (item.contains("captures")) {
        if (auto err = parseCaptures(item["captures"], &rule)) {
          return std::move(err);
        }
      }
      if (item.contains("where")) {
        if (!item["where"].is_array()) {
          return llvm::createStringError(std::errc::invalid_argument,
                                         "pattern '%s' where must be an array",
                                         rule.name.c_str());
        }
        for (const auto &v : item["where"]) {
          if (!v.is_string()) {
            return llvm::createStringError(std::errc::invalid_argument,
                                           "pattern '%s' where entry is not a string",
                                           rule.name.c_str());
          }
          rule.where.push_back(v.get<std::string>());
        }
      }
      config.patterns.push_back(std::move(rule));
    }

    std::sort(config.patterns.begin(), config.patterns.end(),
              [](const PatternRule &a, const PatternRule &b) {
                if (a.priority != b.priority) return a.priority > b.priority;
                if (a.specificity != b.specificity) {
                  return a.specificity > b.specificity;
                }
                return a.name < b.name;
              });
    return config;
  } catch (const std::exception &e) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "invalid fusion config '%s': %s",
                                   path.c_str(), e.what());
  }
}


}  // namespace atir::config_fusion
