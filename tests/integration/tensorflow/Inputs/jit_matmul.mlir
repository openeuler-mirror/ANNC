module {
  func.func @matmul(%lhs: !atir.tensor<2x4x73xf32>,
                    %rhs: !atir.tensor<73x67xf32, name = "weight">,
                    %out: !atir.tensor<2x4x67xf32>) attributes {
      annc.kernel, llvm.emit_c_interface} {
    %dot = atir.Dot %out, %lhs, %rhs : <2x4x67xf32>, <2x4x73xf32>,
        <73x67xf32, name = "weight"> -> <2x4x67xf32>
    return
  }
}
