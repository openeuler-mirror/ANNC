// RUN: annc-asm --atir-fast-codegen %s | FileCheck %s
// The final output shape is [runtime_count, 16]. The fixed embedding width
// must not be mistaken for a Tile multiplier.

// CHECK-LABEL: func.func private @fused_dnn_embedding_hash_bucket
// CHECK: atir.Customize
// CHECK-SAME: custom.op_name = "KPFusedDnnEmbeddingWithHashBucket"
// CHECK-NOT: atir.StringToHashBucketFast
module attributes {module.state = "atir"} {
  func.func private @fused_dnn_embedding_hash_bucket(
      %arg0: !atir.tensor<?xcomplex<f32>, encoding = <"string">, name = "input">,
      %arg1: !atir.tensor<10000x16xf32, name = "embedding_weights">,
      %arg2: !atir.tensor<?x16xf32, name = "output">) attributes {annc.kernel} {
    %0 = "atir.buffer"() : () -> !atir.tensor<?x1xcomplex<f32>, encoding = <"string">, name = "expanded_input">
    %1 = atir.constant "public" @"expand_dim" -> <i32, name = "expand_dim", data = dense<-1> : tensor<i32>>
    %2 = atir.ExpandDims %0, %arg0, %1 : <?x1xcomplex<f32>, encoding = <"string">, name = "expanded_input">, <?xcomplex<f32>, encoding = <"string">, name = "input">, <i32, name = "expand_dim", data = dense<-1> : tensor<i32>> -> <?x1xcomplex<f32>, encoding = <"string">, name = "expanded_input">
    %3 = atir.constant "public" @"ignore_value" -> <complex<f32>, encoding = <"string">, name = "ignore_value", data = strings[""]>
    %4 = atir.Compare %2, %3 {comparisonDirection = "NE"} : (!atir.tensor<?x1xcomplex<f32>, encoding = <"string">, name = "expanded_input">, !atir.tensor<complex<f32>, encoding = <"string">, name = "ignore_value", data = strings[""]>) -> !atir.tensor<?x1xi32, encoding = <"bool">, name = "not_equal">
    %5 = "atir.buffer"() : () -> !atir.tensor<?x2xi64, name = "indices">
    %6 = atir.Where %5, (%4) : <?x2xi64, name = "indices">, !atir.tensor<?x1xi32, encoding = <"bool">, name = "not_equal"> -> <?x2xi64, name = "indices">
    %7 = "atir.buffer"() : () -> !atir.tensor<?xcomplex<f32>, encoding = <"string">, name = "values">
    %8 = atir.GatherNd %7, %2, %6 : <?xcomplex<f32>, encoding = <"string">, name = "values">, <?x1xcomplex<f32>, encoding = <"string">, name = "expanded_input">, <?x2xi64, name = "indices"> -> <?xcomplex<f32>, encoding = <"string">, name = "values">
    %9 = "atir.buffer"() : () -> !atir.tensor<?xi64, name = "hashes">
    %10 = atir.StringToHashBucketFast %9, %8 {numBuckets = 10000 : i64} : <?xi64, name = "hashes">, <?xcomplex<f32>, encoding = <"string">, name = "values"> -> <?xi64, name = "hashes">
    %11, %12 = atir.Unique %10 : (!atir.tensor<?xi64, name = "hashes">) -> (!atir.tensor<?xi64, name = "unique_hashes">, !atir.tensor<?xi64, name = "segment_indices">)
    %13 = "atir.buffer"() : () -> !atir.tensor<?x16xf32, name = "embedding_values">
    %14 = atir.constant "public" @"gather_axis" -> <i32, name = "gather_axis", data = dense<0> : tensor<i32>>
    %15 = atir.Gather %13, %arg1, %11, %14 : <?x16xf32, name = "embedding_values">, <10000x16xf32, name = "embedding_weights">, <?xi64, name = "unique_hashes">, <i32, name = "gather_axis", data = dense<0> : tensor<i32>> -> <?x16xf32, name = "embedding_values">
    %16 = "atir.buffer"() : () -> !atir.tensor<?xi32, name = "segment_ids">
    %17 = atir.Cast %16, %12 : <?xi32, name = "segment_ids">, <?xi64, name = "segment_indices"> -> <?xi32, name = "segment_ids">
    %18 = "atir.buffer"() : () -> !atir.tensor<?x16xf32, name = "segment_mean">
    %19 = atir.constant "private" @"num_segments" -> <i32, name = "num_segments", data = dense<0> : tensor<i32>>
    %20 = atir.SparseSegmentMean %18, %15, %12, %17, %19 : <?x16xf32, name = "segment_mean">, <?x16xf32, name = "embedding_values">, <?xi64, name = "segment_indices">, <?xi32, name = "segment_ids">, <i32, name = "num_segments", data = dense<0> : tensor<i32>> -> <?x16xf32, name = "segment_mean">
    %21 = atir.Shape %20 : (!atir.tensor<?x16xf32, name = "segment_mean">) -> !atir.tensor<2xf32, name = "segment_shape">
    %22 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "shape_as_i32">
    %23 = atir.Cast %22, %21 : <2xi32, name = "shape_as_i32">, <2xf32, name = "segment_shape"> -> <2xi32, name = "shape_as_i32">
    %24 = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "leading_dimension">
    %25 = atir.constant "public" @"leading_begin" -> <1xi32, name = "leading_begin", data = dense<0> : tensor<1xi32>>
    %26 = atir.constant "public" @"leading_size" -> <1xi32, name = "leading_size", data = dense<1> : tensor<1xi32>>
    %27 = atir.Slice %24, %23, %25, %26 : <1xi32, name = "leading_dimension">, <2xi32, name = "shape_as_i32">, <1xi32, name = "leading_begin", data = dense<0> : tensor<1xi32>>, <1xi32, name = "leading_size", data = dense<1> : tensor<1xi32>> -> <1xi32, name = "leading_dimension">
    %28 = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "trailing_dimensions">
    %29 = atir.constant "public" @"trailing_begin" -> <1xi32, name = "trailing_begin", data = dense<1> : tensor<1xi32>>
    %30 = atir.constant "public" @"trailing_size" -> <1xi32, name = "trailing_size", data = dense<-1> : tensor<1xi32>>
    %31 = atir.Slice %28, %23, %29, %30 : <1xi32, name = "trailing_dimensions">, <2xi32, name = "shape_as_i32">, <1xi32, name = "trailing_begin", data = dense<1> : tensor<1xi32>>, <1xi32, name = "trailing_size", data = dense<-1> : tensor<1xi32>> -> <1xi32, name = "trailing_dimensions">
    %32 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "reshaped_shape">
    %33 = atir.constant "public" @"concat_axis" -> <i32, name = "concat_axis", data = dense<0> : tensor<i32>>
    %34 = atir.ConcatV2 %32, %27, %31, %33 : (!atir.tensor<2xi32, name = "reshaped_shape">, !atir.tensor<1xi32, name = "leading_dimension">, !atir.tensor<1xi32, name = "trailing_dimensions">, !atir.tensor<i32, name = "concat_axis", data = dense<0> : tensor<i32>>) -> !atir.tensor<2xi32, name = "reshaped_shape">
    %35 = "atir.buffer"() : () -> !atir.tensor<?x16xf32, name = "reshaped_embedding">
    %36 = atir.Reshape %35, %20, %34 : <?x16xf32, name = "reshaped_embedding">, <?x16xf32, name = "segment_mean">, <2xi32, name = "reshaped_shape"> -> <?x16xf32, name = "reshaped_embedding">
    %37 = atir.Shape %36 : (!atir.tensor<?x16xf32, name = "reshaped_embedding">) -> !atir.tensor<2xf32, name = "output_shape_source">
    %38 = "atir.buffer"() : () -> !atir.tensor<i32, name = "runtime_count">
    %39 = atir.constant "public" @"runtime_begin" -> <1xi32, name = "runtime_begin", data = dense<0> : tensor<1xi32>>
    %40 = atir.constant "public" @"runtime_end" -> <1xi32, name = "runtime_end", data = dense<1> : tensor<1xi32>>
    %41 = atir.constant "public" @"runtime_stride" -> <1xi32, name = "runtime_stride", data = dense<1> : tensor<1xi32>>
    %42 = atir.StridedSlice %38, %37, %39, %40, %41 {shrinkAxisMask = 1 : i32} : <i32, name = "runtime_count">, <2xf32, name = "output_shape_source">, <1xi32, name = "runtime_begin", data = dense<0> : tensor<1xi32>>, <1xi32, name = "runtime_end", data = dense<1> : tensor<1xi32>>, <1xi32, name = "runtime_stride", data = dense<1> : tensor<1xi32>> -> <i32, name = "runtime_count">
    %43 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "output_shape">
    %44 = atir.constant "public" @"embedding_width" -> <i32, name = "embedding_width", data = dense<16> : tensor<i32>>
    %45 = atir.Pack %43, (%42, %44) axis = 0 : <2xi32, name = "output_shape">, !atir.tensor<i32, name = "runtime_count">, !atir.tensor<i32, name = "embedding_width", data = dense<16> : tensor<i32>> -> <2xi32, name = "output_shape">
    %46 = atir.Reshape %arg2, %36, %45 : <?x16xf32, name = "output">, <?x16xf32, name = "reshaped_embedding">, <2xi32, name = "output_shape"> -> <?x16xf32, name = "output">
    return
  }
}
