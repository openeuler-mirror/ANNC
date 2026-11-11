#pragma once

#include <functional>
#include <memory>

#include "common/pb2_common.h"

class GraphPass {
 public:
  virtual ~GraphPass() = default;
  virtual llvm::StringRef getName() const = 0;
  // Returns 0 on success; non-zero is a route exit code (1 IO / 4 invariant).
  virtual int runOnGraph(const tensorflow::GraphDef& graph, BuildCtx& ctx) = 0;
};

// Adapter so stage functions/lambdas become named passes without boilerplate.
class LambdaGraphPass : public GraphPass {
 public:
  using Fn = std::function<int(const tensorflow::GraphDef&, BuildCtx&)>;
  LambdaGraphPass(llvm::StringRef name, Fn fn)
      : name_(name.str()), fn_(std::move(fn)) {}
  llvm::StringRef getName() const override { return name_; }
  int runOnGraph(const tensorflow::GraphDef& graph, BuildCtx& ctx) override {
    return fn_(graph, ctx);
  }

 private:
  std::string name_;
  Fn fn_;
};

inline std::unique_ptr<GraphPass> makeGraphPass(llvm::StringRef name,
                                                LambdaGraphPass::Fn fn) {
  return std::make_unique<LambdaGraphPass>(name, std::move(fn));
}

// Ordered pipeline: runs passes in insertion order, stops at the first
// non-zero code.
class GraphPassManager {
 public:
  void addPass(std::unique_ptr<GraphPass> pass) {
    passes_.push_back(std::move(pass));
  }
  int run(const tensorflow::GraphDef& graph, BuildCtx& ctx) const;

 private:
  std::vector<std::unique_ptr<GraphPass>> passes_;
};
