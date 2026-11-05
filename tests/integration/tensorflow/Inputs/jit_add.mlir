module {
  func.func @add(%lhs: !atir.tensor<1x5xf32>, %rhs: !atir.tensor<1x5xf32>,
                 %out: !atir.tensor<1x5xf32>) attributes {
      annc.kernel, llvm.emit_c_interface} {
    %result = "atir.Add"(%out, %lhs, %rhs) <{do_relu = false, relu_limit = -1.0 : f32}> :
      (!atir.tensor<1x5xf32>, !atir.tensor<1x5xf32>, !atir.tensor<1x5xf32>) -> !atir.tensor<1x5xf32>
    return
  }
}
