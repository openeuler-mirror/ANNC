// RUN: env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 annc-asm %s --convert-context-kernel-abi --split-input-file --verify-diagnostics

module {
  // expected-error @+1 {{context ABI arguments must be builtin ranked tensors}}
  func.func @legacy(%input: !atir.tensor<2x3xf32>) attributes {
    annc.kernel, fusion.metadata = {abi = "annc_context_v3"}
  } {
    return
  }
}

// -----

module {
  // expected-error @+1 {{context ABI arguments must be builtin ranked tensors}}
  func.func @unranked(%input: tensor<*xf32>) attributes {
    annc.kernel, fusion.metadata = {abi = "annc_context_v3"}
  } {
    return
  }
}

// -----

module {
  // expected-error @+1 {{context ABI arguments must be builtin ranked tensors}}
  func.func @memref(%input: memref<2x3xf32>) attributes {
    annc.kernel, fusion.metadata = {abi = "annc_context_v3"}
  } {
    return
  }
}

// -----

module {
  // expected-error @+1 {{context ABI arguments must be builtin ranked tensors}}
  func.func @vector(%input: vector<3xf32>) attributes {
    annc.kernel, fusion.metadata = {abi = "annc_context_v3"}
  } {
    return
  }
}

// -----

module {
  // expected-error @+1 {{context ABI arguments must be builtin ranked tensors}}
  func.func @scalar(%input: f32) attributes {
    annc.kernel, fusion.metadata = {abi = "annc_context_v3"}
  } {
    return
  }
}
