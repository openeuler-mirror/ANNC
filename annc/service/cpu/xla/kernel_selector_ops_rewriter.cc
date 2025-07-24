/* Copyright 2025 Huawei. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "kernel_selector_ops_rewriter.h"

#include <fstream>
#include <regex>
#include <sstream>

#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/cpu_runtime.h"

namespace xla {
namespace cpu {

// Uncomment to get printed information about the sizes and the call selected.
// #define PRINT_DEBUG

#ifdef PRINT_DEBUG
#include <iostream>
#define DEBUG(x) std::cerr << x << "\n";
#else
#define DEBUG(x) \
  do {           \
  } while (0);
#endif

enum Operation { NONE, GEMV, GEMM, BATCH_MATMUL_3D, BATCH_MATMUL_4D };

struct ParsedData {
  std::vector<int> sizes;
  std::string functionName;
};

// Parses line from the mapping file which look like (m,n,k) -> symbol
ParsedData parseLine(const std::string& line) {
  std::regex pattern(R"(\(((\d+,)*\d+)\) -> (.+))");
  ParsedData data;
  std::smatch matches;
  if (std::regex_match(line, matches, pattern)) {
    std::stringstream ss(matches[1]);
    std::string token;
    while (std::getline(ss, token, ',')) {
      data.sizes.push_back(std::stoi(token));
    }
    data.functionName = matches[3];
  } else {
    XLA_VLOG_LINES(3, "KernelSelectorOpsRewriter::parseLine() : No match. \n");
  }
  return data;
}

std::map<std::vector<int>, std::string> sizesToSymbol = {
    // GEMV
    {{256, 256}, runtime::kKernelSelectorGEMVMLIRSymbolName},
    {{128, 256}, runtime::kKernelSelectorGEMVMLIRSymbolName},
    {{256, 64}, runtime::kKernelSelectorGEMVMLIRSymbolName},
    // GEMM
    {{6656, 8, 8}, runtime::kKernelSelectorGEMMMLIRSymbolName},
    {{128, 1024, 416}, runtime::kKernelSelectorGEMMSymbolName},
    {{128, 512, 1024}, runtime::kKernelSelectorGEMMSymbolName},
    {{128, 256, 512}, runtime::kKernelSelectorGEMMSymbolName},
    {{1536, 768, 3072}, runtime::kKernelSelectorGEMMSymbolName},
    {{1536, 21128, 768}, runtime::kKernelSelectorGEMMSymbolName},
    {{1536, 3072, 768}, runtime::kKernelSelectorGEMMSymbolName},
    {{1536, 768, 768}, runtime::kKernelSelectorGEMMSymbolName},
    // BATCH3D
    {{512, 26, 4, 26}, runtime::kKernelSelectorBatch3DMLIRSymbolName},
    {{512, 26, 26, 4}, runtime::kKernelSelectorBatch3DMLIRSymbolName},
    // BATCH4D
    {{4, 12, 384, 64, 384}, runtime::kKernelSelectorBatch4DMLIRSymbolName},
    {{4, 12, 384, 384, 64}, runtime::kKernelSelectorBatch4DMLIRSymbolName}};

const char* kernel_map_file = std::getenv("KERNEL_MAP_FILE");

template <typename T1, typename T2>
void fill_map_from_file(const char* map_file, std::map<T1, T2>& map) {
  if (!map_file) {
    XLA_VLOG_LINES(3, "NO MAP FILE\n");
    return;
  }

  std::ifstream file(map_file);
  if (!file.is_open()) {
    std::string file_name(map_file);
    XLA_VLOG_LINES(3,
                   "KernelSelectorOpsRewriter::fill_map_from_file() : Cannot "
                   "open file. \n");
    return;
  }

  // Clear the map to prevent conflicts and unexpected
  // behaviour due to default pre-filled values.
  map.clear();

  std::string line;
  while (std::getline(file, line)) {
    ParsedData data = parseLine(line);
    if (!data.functionName.empty()) {
      map[data.sizes] = data.functionName;
    }
  }

  return;
}

class KernelSelectorOpsRewriterVisitor : public DfsHloRewriteVisitor {
 private:
  Operation getOperation(HloInstruction* instr) {
    if (auto* dot = DynCast<HloDotInstruction>(instr)) {
      auto batch_dims = dot->dot_dimension_numbers().lhs_batch_dimensions();
      auto dims = dot->shape().dimensions();
      if (batch_dims.size() == 1) {
        return Operation::BATCH_MATMUL_3D;
      }
      if (batch_dims.size() == 2) {
        return Operation::BATCH_MATMUL_4D;
      }
      if (dims.size() == 1) {
        return Operation::GEMV;
      }
      if (batch_dims.empty()) {
        return Operation::GEMM;
      }
    }
    return Operation::NONE;
  }

  template <typename T>
  HloInstruction* makeConstant(HloInstruction* op, T value) {
    auto litteral = LiteralUtil::CreateR0<T>(value);
    return op->AddInstruction(
        HloInstruction::CreateConstant(std::move(litteral)));
  }

#ifdef PRINT_DEBUG
  std::map<std::vector<int>, std::string> AllocatedGemmSizes;
  std::map<std::vector<int>, std::string> AllocatedGemvSizes;
  std::map<std::vector<int>, std::string> AllocatedBatchMatmul3DSizes;
  std::map<std::vector<int>, std::string> AllocatedBatchMatmul4DSizes;
#endif

 public:
  Status HandleDot(HloInstruction* dot) override {
    Operation operation = getOperation(dot);
    if (operation == Operation::NONE) {
      return OkStatus();
    }

    // Collect all the operands for the CustomCall
    switch (operation) {
      case GEMM: {
        auto dnums = dot->dot_dimension_numbers();
        auto lhs_contracting_dims = dnums.lhs_contracting_dimensions();
        auto rhs_contracting_dims = dnums.rhs_contracting_dimensions();

        assert(lhs_contracting_dims.size() == 1);
        assert(rhs_contracting_dims.size() == 1);

        HloInstruction* trA = makeConstant(dot, lhs_contracting_dims[0] == 0);
        HloInstruction* trB = makeConstant(dot, rhs_contracting_dims[0] == 1);

        HloInstruction* alpha = makeConstant(dot, (float)1.0);
        HloInstruction* beta = makeConstant(dot, (float)0.0);

        HloInstruction* A = dot->operands()[0];
        HloInstruction* B = dot->operands()[1];

        int m = dot->shape().dimensions(0);
        HloInstruction* M = makeConstant(dot, m);

        int n = dot->shape().dimensions(1);
        HloInstruction* N = makeConstant(dot, n);

        int k = A->shape().dimensions(lhs_contracting_dims[0]);
        HloInstruction* K = makeConstant(dot, k);

        if (sizesToSymbol.find({m, n, k}) == sizesToSymbol.end()) {
#ifdef PRINT_DEBUG
          DEBUG("{m: " << m << ", n: " << n << ", k: " << k << "} -> "
                       << "Is not on the map. The dot will not be replaced.");
#endif
          return OkStatus();
        }

        auto fun_name = sizesToSymbol[{m, n, k}];

#ifdef PRINT_DEBUG
        if (AllocatedGemmSizes.find({m, n, k}) == AllocatedGemmSizes.end()) {
          AllocatedGemmSizes[{m, n, k}] = fun_name;
          DEBUG("{m: " << m << ", n: " << n << ", k: " << k << "} -> "
                       << fun_name);
        }
#endif

        std::vector<HloInstruction*> operands = {trA, trB, A,     B,   M,
                                                 N,   K,   alpha, beta};

        HloInstruction* kernel_selector_call =
            dot->AddInstruction(HloInstruction::CreateCustomCall(
                dot->shape(), operands, "KernelSelector"));

        // Add metadata
        OpMetadata metadata = dot->metadata();
        metadata.set_op_name(fun_name);
        metadata.set_op_type(runtime::kKernelSelectorOperationGEMM);
        kernel_selector_call->set_metadata(metadata);
        TF_RETURN_IF_ERROR(ReplaceInstruction(dot, kernel_selector_call));

        break;
      }
      case GEMV: {
        auto dnums = dot->dot_dimension_numbers();
        auto lhs_contracting_dims = dnums.lhs_contracting_dimensions();

        assert(lhs_contracting_dims.size() == 1);

        bool is_trA = lhs_contracting_dims[0] == 0;
        HloInstruction* trA = makeConstant(dot, is_trA);

        HloInstruction* alpha = makeConstant(dot, (float)1.0);
        HloInstruction* beta = makeConstant(dot, (float)0.0);

        HloInstruction* A = dot->operands()[0];
        HloInstruction* X = dot->operands()[1];

        int m = A->shape().dimensions(is_trA ? 1 : 0);
        HloInstruction* M = makeConstant(dot, m);

        int n = A->shape().dimensions(is_trA ? 0 : 1);
        HloInstruction* N = makeConstant(dot, n);

        // If (m,n) is not in the map, do not do anything.
        if (sizesToSymbol.find({m, n}) == sizesToSymbol.end()) {
#ifdef PRINT_DEBUG
          DEBUG("{m: " << m << ", n: " << n << "} -> "
                       << "Is not on the map. The dot will not be replaced.");
#endif
          return OkStatus();
        }

        auto fun_name = sizesToSymbol[{m, n}];

#ifdef PRINT_DEBUG
        if (AllocatedGemvSizes.find({m, n}) == AllocatedGemvSizes.end()) {
          AllocatedGemvSizes[{m, n}] = fun_name;
          DEBUG("{m: " << m << ", n: " << n << "} -> " << fun_name);
        }
#endif

        std::vector<HloInstruction*> operands = {trA, A, X, M, N, alpha, beta};

        HloInstruction* kernel_selector_call =
            dot->AddInstruction(HloInstruction::CreateCustomCall(
                dot->shape(), operands, "KernelSelector"));

        // Add metadata
        OpMetadata metadata = dot->metadata();
        metadata.set_op_name(fun_name);
        metadata.set_op_type(runtime::kKernelSelectorOperationGEMV);
        kernel_selector_call->set_metadata(metadata);
        TF_RETURN_IF_ERROR(ReplaceInstruction(dot, kernel_selector_call));

        break;
      }
      case BATCH_MATMUL_3D: {
        auto dnums = dot->dot_dimension_numbers();
        auto lhs_contracting_dims = dnums.lhs_contracting_dimensions();
        auto rhs_contracting_dims = dnums.rhs_contracting_dimensions();

        assert(lhs_contracting_dims.size() == 1);
        assert(rhs_contracting_dims.size() == 1);

        HloInstruction* trA = makeConstant(dot, lhs_contracting_dims[0] == 1);
        HloInstruction* trB = makeConstant(dot, rhs_contracting_dims[0] == 2);

        HloInstruction* A = dot->operands()[0];
        HloInstruction* B = dot->operands()[1];

        int p = dot->shape().dimensions(0);
        HloInstruction* P = makeConstant(dot, p);

        int num_batch_dims = dnums.lhs_batch_dimensions_size();

        int m = dot->shape().dimensions(num_batch_dims);
        HloInstruction* M = makeConstant(dot, m);

        int n = dot->shape().dimensions(num_batch_dims + 1);
        HloInstruction* N = makeConstant(dot, n);

        int k = A->shape().dimensions(lhs_contracting_dims[0]);
        HloInstruction* K = makeConstant(dot, k);

        // If (p,m,n,k) is not in the map, do not do anything.
        if (sizesToSymbol.find({p, m, n, k}) == sizesToSymbol.end()) {
#ifdef PRINT_DEBUG
          DEBUG("{p: " << p << ", m: " << m << ", n: " << n << ", k: " << k
                       << "} -> "
                       << "  Is not on the map. The dot will not be replaced.");
#endif
          return OkStatus();
        }

        auto fun_name = sizesToSymbol[{p, m, n, k}];

#ifdef PRINT_DEBUG
        if (AllocatedBatchMatmul3DSizes.find({p, m, n, k}) ==
            AllocatedBatchMatmul3DSizes.end()) {
          AllocatedBatchMatmul3DSizes[{p, m, n, k}] = fun_name;
          DEBUG("{p: " << p << ", m: " << m << ", n: " << n << ", k: " << k
                       << "} -> " << fun_name);
        }
#endif

        std::vector<HloInstruction*> operands = {trA, trB, A, B, P, M, N, K};

        HloInstruction* kernel_selector_call =
            dot->AddInstruction(HloInstruction::CreateCustomCall(
                dot->shape(), operands, "KernelSelector"));

        // Add metadata
        OpMetadata metadata = dot->metadata();
        metadata.set_op_name(fun_name);
        metadata.set_op_type(runtime::kKernelSelectorOperationBATCH3D);
        kernel_selector_call->set_metadata(metadata);
        TF_RETURN_IF_ERROR(ReplaceInstruction(dot, kernel_selector_call));

        break;
      }
      case BATCH_MATMUL_4D: {
        auto dnums = dot->dot_dimension_numbers();
        auto lhs_contracting_dims = dnums.lhs_contracting_dimensions();
        auto rhs_contracting_dims = dnums.rhs_contracting_dimensions();

        assert(lhs_contracting_dims.size() == 1);
        assert(rhs_contracting_dims.size() == 1);

        HloInstruction* trA = makeConstant(dot, lhs_contracting_dims[0] == 2);
        HloInstruction* trB = makeConstant(dot, rhs_contracting_dims[0] == 3);

        HloInstruction* A = dot->operands()[0];
        HloInstruction* B = dot->operands()[1];

        int q = dot->shape().dimensions(0);
        HloInstruction* Q = makeConstant(dot, q);

        int p = dot->shape().dimensions(1);
        HloInstruction* P = makeConstant(dot, p);

        int num_batch_dims = dnums.lhs_batch_dimensions_size();

        int m = dot->shape().dimensions(num_batch_dims);
        HloInstruction* M = makeConstant(dot, m);

        int n = dot->shape().dimensions(num_batch_dims + 1);
        HloInstruction* N = makeConstant(dot, n);

        int k = A->shape().dimensions(lhs_contracting_dims[0]);
        HloInstruction* K = makeConstant(dot, k);

        // If (q,p,m,n,k) is not in the map, do not do anything.
        if (sizesToSymbol.find({q, p, m, n, k}) == sizesToSymbol.end()) {
#ifdef PRINT_DEBUG
          DEBUG("{q: " << q << ", p: " << p << ", m: " << m << ", n: " << n
                       << ", k: " << k << "} -> "
                       << "Is not on the map. The dot will not be replaced.");
#endif
          return OkStatus();
        }

        auto fun_name = sizesToSymbol[{q, p, m, n, k}];

#ifdef PRINT_DEBUG
        if (AllocatedBatchMatmul4DSizes.find({q, p, m, n, k}) ==
            AllocatedBatchMatmul4DSizes.end()) {
          AllocatedBatchMatmul4DSizes[{q, p, m, n, k}] = fun_name;
          DEBUG("{q: " << q << ", p: " << p << ", m: " << m << ", n: " << n
                       << ", k: " << k << "} -> " << fun_name);
        }
#endif

        std::vector<HloInstruction*> operands = {trA, trB, A, B, Q, P, M, N, K};

        HloInstruction* kernel_selector_call =
            dot->AddInstruction(HloInstruction::CreateCustomCall(
                dot->shape(), operands, "KernelSelector"));

        // Add metadata
        OpMetadata metadata = dot->metadata();
        metadata.set_op_name(fun_name);
        metadata.set_op_type(runtime::kKernelSelectorOperationBATCH4D);
        kernel_selector_call->set_metadata(metadata);
        TF_RETURN_IF_ERROR(ReplaceInstruction(dot, kernel_selector_call));

        break;
      }
      default:
        DEBUG("No library funcion was selected.");
        return OkStatus();
    }

    return OkStatus();
  }
};  // namespace cpu

absl::StatusOr<bool> KernelSelectorOpsRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      3, "KernelSelectorOpsRewriter::Run(), before:\n" + module->ToString());

  fill_map_from_file(kernel_map_file, sizesToSymbol);

  KernelSelectorOpsRewriterVisitor visitor;
  TF_ASSIGN_OR_RETURN(auto result,
                      visitor.RunOnModule(module, execution_threads));
  XLA_VLOG_LINES(
      3, "KernelSelectorOpsRewriter::Run(), after:\n" + module->ToString());
  return result;
}

}  // namespace cpu
}  // namespace xla
