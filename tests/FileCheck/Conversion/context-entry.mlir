// RUN: split-file %s %t
// RUN: annc-asm %t/legacy.mlir --lower-context-kernel-entries | FileCheck %s
// RUN: annc-asm %t/invalid.mlir --lower-context-kernel-entries --split-input-file --verify-diagnostics

// The standalone pass preserves ordinary functions and both legacy ABIs.
// CHECK-LABEL: func.func @ordinary(
// CHECK-SAME: %{{.*}}: i64) -> i64
// CHECK: return %{{.*}} : i64
// CHECK-LABEL: func.func @ciface(
// CHECK-SAME: %{{.*}}: i64) -> i64
// CHECK: return %{{.*}} : i64
// CHECK-LABEL: func.func @execution_v2(
// CHECK-SAME: %{{.*}}: i64) -> i64
// CHECK: return %{{.*}} : i64

//--- legacy.mlir
module {
  func.func @ordinary(%value: i64) -> i64 {
    return %value : i64
  }
  func.func @ciface(%value: i64) -> i64 attributes {
    fusion.metadata = {abi = "mlir_ciface"}
  } {
    return %value : i64
  }
  func.func @execution_v2(%value: i64) -> i64 attributes {
    fusion.metadata = {abi = "annc_execution_v2"}
  } {
    return %value : i64
  }
}

//--- invalid.mlir
module {
  // expected-error @+1 {{context entry must be a single-block kernel}}
  func.func @not_a_kernel() attributes {
    fusion.metadata = {abi = "annc_context_v3"}
  } {
    return
  }
}
// -----
module {
  // expected-error @+1 {{context entry requires builtin inputs and outputs in boundary order}}
  func.func @missing_builtin() attributes {
    annc.kernel, fusion.metadata = {abi = "annc_context_v3"}
  } {
    return
  }
}
// -----
module {
  // expected-error @+1 {{unsupported fusion ABI: annc_context_v2}}
  func.func @unknown_abi() attributes {
    fusion.metadata = {abi = "annc_context_v2"}
  } {
    return
  }
}
