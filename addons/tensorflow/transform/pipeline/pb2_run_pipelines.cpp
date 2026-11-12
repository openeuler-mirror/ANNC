#include <algorithm>
#include <utility>
#include <vector>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "common/pb2_common.h"
#include "passes/pb2_mlir_passes.h"

// Official preprocess (unless skipped) + tf->stablehlo pipeline, then wrap
// registered leftovers as tf.residue and emit the stats report.
int RunPipelines(const Options& opt, const BuildCtx& ctx,
                 const tensorflow::GraphDef& graph, mlir::MLIRContext& context,
                 mlir::ModuleOp& module) {
  std::optional<tensorflow::Session*> no_session;
  // Only freeze shapes when EVERY entry has one: an entry without a shape
  // would arrive as an empty dims list, which TF hard-freezes as a rank-0
  // scalar (no unranked spelling exists on this channel) -- silently
  // corrupting the signature. Unshaped entries skip freezing entirely.
  llvm::SmallVector<std::vector<int64_t>> arg_shapes_hold;
  llvm::SmallVector<llvm::ArrayRef<int64_t>> arg_shapes;
  bool allShaped = true;
  for (auto& s : opt.inputs) {
    std::string name = s.substr(0, s.find(':'));
    auto it = ctx.entries.find(name);
    if (it == ctx.entries.end()) {
      std::cerr << "[FATAL] entry " << name
                << " has no EntrySpec (--input parsing inconsistent with "
                   "signature)\n";
      return 1;
    }
    if (!it->second.has_shape) allShaped = false;
    arg_shapes_hold.emplace_back(it->second.dims.begin(),
                                 it->second.dims.end());
    arg_shapes.push_back(arg_shapes_hold.back());
  }
  if (!allShaped) {
    arg_shapes.clear();
    std::cerr << "[pipeline] unshaped entry present: shape freezing skipped "
                 "(an empty dims entry would freeze as rank-0)\n";
  }
  // Dump prefix only under an explicit crash-repro request: a fixed
  // world-writable path is a symlink hazard on shared machines.
  std::string dump_prefix;
  if (!opt.crash_repro_path.empty()) {
    auto slash = opt.crash_repro_path.find_last_of('/');
    dump_prefix =
        (slash == std::string::npos ? std::string(".")
                                    : opt.crash_repro_path.substr(0, slash)) +
        "/pre_";
  }
  if (!opt.skip_preprocess) {
    auto pstatus = tensorflow::quantization::PreprocessAndFreezeGraph(
        /*mlir_dump_file_prefix=*/dump_prefix,
        /*is_inliner_run=*/false,
        /*noinline_functions=*/{},
        /*module_op=*/module,
        /*context=*/&context,
        /*session=*/no_session,
        /*run_tf_to_stablehlo=*/false,
        /*deserialize_xla_call_module=*/false,
        /*input_arg_shapes=*/
        llvm::ArrayRef<llvm::ArrayRef<int64_t>>(arg_shapes));
    if (!pstatus.ok()) {
      std::cerr << "[FAIL] PreprocessAndFreezeGraph: " << pstatus.ToString()
                << "\n";
      return 1;
    }
  } else {
    std::cerr
        << "[pipeline] skipped official preprocess (Smuggle guard active)\n";
  }
  mlir::PassManager pm(&context);
  if (!opt.crash_repro_path.empty())
    pm.enableCrashReproducerGeneration(opt.crash_repro_path);
  pm.enableVerifier(false);
  shlo_pipeline::AddTFToStablehloPasses(pm);
  // Post-legalize unification: remaining registered tf.* ops wrapped as
  // tf.residue (sink F1 dispatches by the attr; collect then rewrite).
  pm.addPass(pb_passes::CreateNormalizeShapeVectorsPass());
  pm.addPass(pb_passes::CreateWrapRegisteredResiduePass());
  if (failed(pm.run(module))) {
    std::cerr << "[FAIL] tf->stablehlo legalize pipeline failed\n";
    return 1;
  }
  std::cerr << "[pipeline] tf->stablehlo OK (residues carried as annc.residue)\n";
  module.walk([&](mlir::Operation* op) {
    if (op->hasAttr("result_accuracy")) op->removeAttr("result_accuracy");
  });
  {
    mlir::PassManager mpm(&context);
    mpm.enableVerifier(false);
    mpm.addPass(pb_passes::CreateTagConstSourceNamePass());
    mpm.addPass(pb_passes::CreateRestoreConstSourceLocPass(graph));
    mpm.addPass(pb_passes::CreateAnchorConstantsToUsersPass());
    // Port tags run after the universal source_node attribution: the tag is
    // derived from group membership + the consumer's pb input spec, both of
    // which are only complete at this point. The consumption graph is final.
    mpm.addPass(pb_passes::CreatePortTagPass(graph));
    if (failed(mpm.run(module))) {
      std::cerr << "[FATAL] anchor pass failed\n";
      return 1;
    }
  }
  // Carrier rename strictly last: the source-loc passes above identify TF
  // carrier hosts by the tf.residue name, so the rename must not run
  // between wrap and them. annc.residue is what the sink finally sees.
  {
    mlir::PassManager rpm(&context);
    rpm.enableVerifier(false);
    rpm.addPass(pb_passes::CreateRenameResidueCarrierPass());
    if (failed(rpm.run(module))) {
      std::cerr << "[FATAL] carrier rename failed\n";
      return 1;
    }
  }
  if (!opt.out_path.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream out(opt.out_path, ec, llvm::sys::fs::OF_None);
    if (ec) {
      std::cerr << "[FATAL] cannot open output file: " << opt.out_path
                << " (" << ec.message() << ")\n";
      return 1;
    }
    module.print(out, mlir::OpPrintingFlags().enableDebugInfo());
    out.flush();
  } else if (opt.dump_out) {
    module.print(llvm::outs(), mlir::OpPrintingFlags().enableDebugInfo());
    llvm::outs().flush();
  }
  return 0;
}
