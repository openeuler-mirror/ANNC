#include "ConfigFusionSupport.h"

#include <system_error>

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "nlohmann/json.hpp"

using namespace llvm;
using namespace mlir;

namespace atir::config_fusion {

StringRef stringifySkipReason(SkipReason reason) {
  switch (reason) {
    case SkipReason::ConfigurationError:
      return "configuration_error";
    case SkipReason::StructureMismatch:
      return "structure_mismatch";
    case SkipReason::ConstraintMismatch:
      return "constraint_mismatch";
    case SkipReason::CaptureFailure:
      return "capture_failure";
    case SkipReason::InputMismatch:
      return "input_mismatch";
    case SkipReason::BoundaryEscape:
      return "boundary_escape";
    case SkipReason::MaterializationFailure:
      return "materialization_failure";
  }
  llvm_unreachable("unknown config fusion skip reason");
}

static std::string describeAnchor(Operation *anchor) {
  if (!anchor) return "<configuration>";
  std::string description = simpleOpName(anchor);
  std::string location;
  raw_string_ostream os(location);
  anchor->getLoc().print(os);
  if (!location.empty() && location != "unknown") {
    description += " at ";
    description += location;
  }
  return description;
}

static std::string printOperation(Operation *operation) {
  if (!operation) return "";
  std::string ir;
  raw_string_ostream os(ir);
  operation->print(os, OpPrintingFlags().skipRegions());
  return ir;
}

void PatternReport::recordAnchor(Operation *anchor) {
  if (anchor && seenAnchors.insert(anchor).second) ++anchors;
}

void PatternReport::recordSkip(SkipReason reason, Operation *anchor,
                               StringRef message, int64_t sampleLimit,
                               StringRef statement, Operation *mismatch) {
  if (!seenSkips[anchor].insert(reason).second) return;

  std::string reasonName = stringifySkipReason(reason).str();
  ++skips[reasonName];
  if (sampleLimit <= 0 || static_cast<int64_t>(samples.size()) >= sampleLimit) {
    return;
  }
  samples.push_back({reasonName, message.str(), describeAnchor(anchor),
                     statement.str(), printOperation(anchor),
                     mismatch != anchor ? printOperation(mismatch) : ""});
}

void emitPatternWarnings(func::FuncOp mainFunc,
                         const std::vector<PatternReport> &reports) {
  for (const PatternReport &report : reports) {
    if (report.skips.empty()) continue;

    int64_t total = 0;
    for (const auto &[reason, count] : report.skips) total += count;
    auto diagnostic = mainFunc.emitWarning();
    if (report.skips.count("configuration_error")) {
      diagnostic << "config fusion pattern '" << report.name << "' is invalid";
    } else {
      diagnostic << "config fusion pattern '" << report.name << "' skipped "
                 << total << " candidate(s)";
    }
    for (const auto &[reason, count] : report.skips) {
      diagnostic << " " << reason << "=" << count;
    }
    for (const SkipSample &sample : report.samples) {
      mlir::Diagnostic &note = diagnostic.attachNote();
      note << sample.reason << ": " << sample.message;
      if (!sample.statement.empty()) {
        note << " [statement: " << sample.statement << "]";
      }
      note << " [anchor: " << sample.anchor << "]";
    }
  }
}

Error writeFusionReport(StringRef path, StringRef configPath,
                        const std::vector<PatternReport> &reports) {
  nlohmann::json root;
  root["version"] = 1;
  root["config"] = configPath.str();
  root["patterns"] = nlohmann::json::array();

  for (const PatternReport &report : reports) {
    nlohmann::json pattern;
    pattern["name"] = report.name;
    pattern["status"] = report.skips.count("configuration_error") ? "invalid"
                                                                   : "active";
    pattern["anchors"] = report.anchors;
    pattern["matches"] = report.matches;
    pattern["skips"] = nlohmann::json::object();
    for (const auto &[reason, count] : report.skips) {
      pattern["skips"][reason] = count;
    }
    pattern["samples"] = nlohmann::json::array();
    for (const SkipSample &sample : report.samples) {
      nlohmann::json sampleJson = {{"reason", sample.reason},
                                   {"message", sample.message},
                                   {"anchor", sample.anchor}};
      if (!sample.statement.empty()) {
        sampleJson["statement"] = sample.statement;
      }
      if (!sample.ir.empty()) {
        sampleJson["ir"] = sample.ir;
      }
      if (!sample.mismatchIr.empty()) {
        sampleJson["mismatch_ir"] = sample.mismatchIr;
      }
      pattern["samples"].push_back(std::move(sampleJson));
    }
    pattern["outlined"] = nlohmann::json::array();
    for (const std::string &outlined : report.outlined) {
      pattern["outlined"].push_back(outlined);
    }
    pattern["inferred_roles"] = nlohmann::json::object();
    for (const auto &[input, role] : report.inferredRoles) {
      pattern["inferred_roles"][input] = role;
    }
    root["patterns"].push_back(std::move(pattern));
  }

  std::error_code ec;
  raw_fd_ostream output(path, ec, sys::fs::OF_Text);
  if (ec) {
    return createStringError(ec, "cannot open config fusion report '%s'",
                             path.str().c_str());
  }
  output << root.dump(2) << "\n";
  output.flush();
  if (output.has_error()) {
    return createStringError(std::errc::io_error,
                             "failed to write config fusion report '%s'",
                             path.str().c_str());
  }
  return Error::success();
}

}  // namespace atir::config_fusion
