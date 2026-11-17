// RUN: annc-asm %s --split-input-file --verify-diagnostics

func.func @missing(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>, %bias: tensor<4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{requires annc.gemm.epilogue when epilogue inputs are present}}
  %0 = atir.fused_matmul %lhs, %rhs, %bias : (tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @wrong_type(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{requires annc.gemm.epilogue to be a dictionary}}
  %0 = atir.fused_matmul %lhs, %rhs {annc.gemm.epilogue = "invalid"} : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @missing_steps(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{requires a nonempty epilogue steps array}}
  %0 = atir.fused_matmul %lhs, %rhs {annc.gemm.epilogue = {}} : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @missing_input(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{epilogue inputs must match the ordered binary steps}}
  %0 = atir.fused_matmul %lhs, %rhs {annc.gemm.epilogue = {steps = [{opcode = "add"}]}} : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @extra_input(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>, %bias: tensor<4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{epilogue inputs must match the ordered binary steps}}
  %0 = atir.fused_matmul %lhs, %rhs, %bias {annc.gemm.epilogue = {steps = [{opcode = "relu"}]}} : (tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @unknown_opcode(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{has an unsupported epilogue opcode}}
  %0 = atir.fused_matmul %lhs, %rhs {annc.gemm.epilogue = {steps = [{opcode = "sub"}]}} : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
