// RUN: annc-opt %s --atir-prune-func=execution-mode=aot | FileCheck %s

module {
  func.func private @aot_kernel() attributes {
    annc.kernel,
    annc.execution_mode = "aot",
    fusion.pattern = "embedding"
  } {
    return
  }

  func.func private @jit_kernel() attributes {
    annc.kernel,
    annc.execution_mode = "jit",
    fusion.pattern = "matmul"
  } {
    return
  }
}

// CHECK-LABEL: func.func private @aot_kernel
// CHECK-NOT: @jit_kernel
