#include "JitPrepack.h"

#include <limits>
#include <set>

#include "../GemmPlan.h"
#include "Kernel/PackedRhs.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

// The helper is supplied by the optional AArch64 GEMM microkernel archive.
// Keep the compiler build usable without that archive; reject SVE prepacking
// below when the sizing contract cannot be checked.
extern "C" int64_t annc_aarch64_sve_packed_b_elements_f32(int64_t k, int64_t n)
    __attribute__((weak));

namespace annc {
using namespace mlir;
using namespace aarch64::gemm;

LogicalResult prepareJitRhs(ModuleOp module, llvm::StringRef manifest,
                            llvm::StringRef output) {
  auto file = llvm::MemoryBuffer::getFile(manifest);
  if (!file)
    return module.emitError("cannot read JIT immutable input manifest");
  auto json = llvm::json::parse((*file)->getBuffer());
  if (!json) {
    llvm::consumeError(json.takeError());
    return module.emitError("invalid JIT immutable input manifest JSON");
  }
  auto* entries = json->getAsArray();
  if (!entries)
    return module.emitError("JIT immutable input manifest must be an array");
  size_t functions = 0;
  for (auto func : module.getOps<func::FuncOp>())
    if (!func.isDeclaration()) ++functions;
  if (functions != 1)
    return module.emitError(
        "JIT immutable input manifest requires one selected function");
  std::set<int64_t> inputs;
  for (auto& entry : *entries) {
    auto argument = entry.getAsInteger();
    if (!argument || *argument < 0 || !inputs.insert(*argument).second)
      return module.emitError("invalid/duplicate JIT immutable argument");
  }
  std::error_code error;
  llvm::raw_fd_ostream out(output, error);
  if (error) return module.emitError("cannot write JIT packing source");
  out << "#include <stdint.h>\n";
  out << "typedef struct { int64_t version, argument, k, n, kc, nc, isa, "
         "panel, elements, alignment; } Spec;\n";
  out << "extern void annc_aarch64_neon_packb_f32(const "
         "float*,float*,int32_t,int32_t,int32_t);\n";
  out << "extern void annc_aarch64_sve_packb_f32(const "
         "float*,float*,int32_t,int32_t,int32_t);\n";
  std::vector<AnncPackedRhsSpec> specs;
  bool failedPack = false;
  OpBuilder builder(module.getContext());
  module.walk([&](func::FuncOp func) {
    func.walk([&](Operation* op) {
      if (!isGemmAnchor(op) || !op->hasAttr(kCandidateAttrName)) return;
      auto candidate = readCandidate(op);
      if (failed(candidate)) {
        failedPack = true;
        return;
      }
      if (candidate->strategy.rhsPacking != RhsPacking::kPrepacked) return;
      Value rhs = getGemmInput(op, 1), root = rhs;
      bool exclusive = root.hasOneUse();
      while (auto cast = root.getDefiningOp<memref::CastOp>()) {
        root = cast.getSource();
        exclusive &= root.hasOneUse();
      }
      auto argument = dyn_cast<BlockArgument>(root);
      auto type = dyn_cast<MemRefType>(rhs.getType());
      bool eligible =
          exclusive && argument && argument.getOwner() == &func.front() &&
          type && type.getRank() == 2 && type.hasStaticShape() &&
          type.getLayout().isIdentity() && type.getElementType().isF32() &&
          inputs.count(argument.getArgNumber());
      if (!eligible) {
        NamedAttrList attrs(
            op->getAttrOfType<DictionaryAttr>(kCandidateAttrName));
        attrs.set("rhs_packing", builder.getStringAttr("packed"));
        op->setAttr(kCandidateAttrName,
                    attrs.getDictionary(module.getContext()));
        return;
      }
      const auto& strategy = candidate->strategy;
      const auto& abi =
          getGemmKernelABI(strategy.target, strategy.isa, strategy.dataType,
                           strategy.executionKind);
      AnncPackedRhsSpec s{
          1,
          argument.getArgNumber(),
          type.getShape()[0],
          type.getShape()[1],
          strategy.cacheTile.kc,
          strategy.cacheTile.nc,
          strategy.isa == GemmIsa::kSve ? 1 : 0,
          strategy.isa == GemmIsa::kSve ? abi.vectorLengthBytes : 4,
          0,
          64};
      if (s.k <= 0 || s.n <= 0 || s.k > INT32_MAX || s.n > INT32_MAX) {
        op->emitError("unsupported JIT RHS dimensions");
        failedPack = true;
        return;
      }
      if (s.isa && !annc_aarch64_sve_packed_b_elements_f32) {
        op->emitError(
            "JIT SVE RHS prepacking requires the AArch64 GEMM microkernel "
            "archive");
        failedPack = true;
        return;
      }
      if (s.isa && annc_aarch64_sve_packed_b_elements_f32(1, 1) != s.panel) {
        op->emitError("JIT packed RHS SVE vector length does not match target");
        failedPack = true;
        return;
      }
      auto blocks = partitionCache2D(s.k, s.n, s.kc, s.nc);
      for (const auto& b : blocks) {
        const int64_t width = ((b.nSize + s.panel - 1) / s.panel) * s.panel;
        if (b.kSize >
            (std::numeric_limits<int64_t>::max() - s.elements) / width) {
          op->emitError("JIT packed RHS size overflow");
          failedPack = true;
          return;
        }
        if (s.isa && annc_aarch64_sve_packed_b_elements_f32(b.kSize, b.nSize) !=
                         b.kSize * width) {
          op->emitError("JIT packed RHS SVE sizing does not match target");
          failedPack = true;
          return;
        }
        s.elements += b.kSize * width;
      }
      if (uint64_t(s.elements) >
          std::numeric_limits<size_t>::max() / sizeof(float)) {
        op->emitError("JIT packed RHS byte size overflow");
        failedPack = true;
        return;
      }
      const size_t index = specs.size();
      std::string symbol = "annc_jit_packed_" + std::to_string(index);
      out << "static void pack_" << index
          << "(const float* src,float* dst) {\n";
      int64_t offset = 0;
      for (const auto& b : blocks) {
        out << "annc_aarch64_" << (s.isa ? "sve" : "neon") << "_packb_f32(src+"
            << b.kStart * s.n + b.nStart << ",dst+" << offset << "," << b.kSize
            << "," << b.nSize << "," << s.n << ");\n";
        offset += b.kSize * ((b.nSize + s.panel - 1) / s.panel) * s.panel;
      }
      out << "}\n";
      NamedAttrList contract;
      contract.set("version", builder.getI64IntegerAttr(kPrepackedRhsVersion));
      contract.set("k", builder.getI64IntegerAttr(s.k));
      contract.set("n", builder.getI64IntegerAttr(s.n));
      contract.set("kc", builder.getI64IntegerAttr(s.kc));
      contract.set("nc", builder.getI64IntegerAttr(s.nc));
      contract.set("data_elements", builder.getI64IntegerAttr(s.elements));
      contract.set("data_symbol", builder.getStringAttr(symbol));
      contract.set("input", builder.getUnitAttr());
      op->setAttr(kPrepackedRhsAttrName,
                  contract.getDictionary(module.getContext()));
      specs.push_back(s);
    });
  });
  if (failedPack) return failure();
  out << "static const Spec specs[]={\n";
  if (specs.empty()) out << "{0},\n";
  for (const auto& s : specs)
    out << "{" << s.version << "," << s.argument << "," << s.k << "," << s.n
        << "," << s.kc << "," << s.nc << "," << s.isa << "," << s.panel << ","
        << s.elements << "," << s.alignment << "},\n";
  out << "};\nconst Spec* annc_jit_packed_rhs_v1(int64_t* count){*count="
      << specs.size() << ";return specs;}\n";
  out << "void annc_jit_prepare_rhs_v1(int64_t index,const float* src,float* "
         "dst){switch(index){\n";
  for (size_t i = 0; i < specs.size(); ++i)
    out << "case " << i << ":pack_" << i << "(src,dst);return;\n";
  out << "}}\n";
  out.flush();
  if (out.has_error())
    return module.emitError("failed to write JIT packing source");
  return success();
}
}  // namespace annc
