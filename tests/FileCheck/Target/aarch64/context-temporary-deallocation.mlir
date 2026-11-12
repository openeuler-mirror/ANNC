// RUN: env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 annc-asm %s --annc-one-shot-bufferize --verify-each | FileCheck %s --check-prefix=OSB
// RUN: env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 annc-asm %s --annc-one-shot-bufferize --verify-each | mlir-opt --buffer-deallocation-pipeline --verify-each | FileCheck %s --check-prefix=DEALLOC

module {
  func.func private @consume(memref<16384xf32>)
  func.func private @consume_large(memref<16385xf32>)
  func.func private @consume_dynamic(memref<?xf32>)
  func.func private @consume_loop(memref<4xf32>)
  // OSB-LABEL: func.func @temporary_buffers(
  // OSB: memref.alloca() : memref<16384xf32>
  // OSB: memref.alloc() : memref<16385xf32>
  // OSB: memref.alloc(%{{.*}}) : memref<?xf32>
  // OSB: scf.for
  // OSB: memref.alloc() : memref<4xf32>
  // DEALLOC-LABEL: func.func @temporary_buffers(
  // DEALLOC: %[[LARGE:.*]] = memref.alloc() : memref<16385xf32>
  // DEALLOC: %[[DYNAMIC:.*]] = memref.alloc(%{{.*}}) : memref<?xf32>
  // DEALLOC: scf.for
  // DEALLOC: %[[LOOP:.*]] = memref.alloc() : memref<4xf32>
  // DEALLOC: memref.dealloc %[[LOOP]] : memref<4xf32>
  // DEALLOC: memref.dealloc %[[LARGE]] : memref<16385xf32>
  // DEALLOC: memref.dealloc %[[DYNAMIC]] : memref<?xf32>
  func.func @temporary_buffers(%context: !llvm.ptr, %input: memref<?xf32>)
      attributes {annc.kernel, fusion.metadata = {abi = "annc_context_v3"}} {
    %zero = arith.constant 0 : index
    %n = memref.dim %input, %zero : memref<?xf32>
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %value = arith.constant 1.0 : f32
    %small = memref.alloc() : memref<16384xf32>
    %large = memref.alloc() : memref<16385xf32>
    %dynamic = memref.alloc(%n) : memref<?xf32>
    memref.store %value, %small[%zero] : memref<16384xf32>
    memref.store %value, %large[%zero] : memref<16385xf32>
    memref.store %value, %dynamic[%zero] : memref<?xf32>
    func.call @consume(%small) : (memref<16384xf32>) -> ()
    func.call @consume_large(%large) : (memref<16385xf32>) -> ()
    func.call @consume_dynamic(%dynamic) : (memref<?xf32>) -> ()
    scf.for %i = %zero to %two step %one {
      %loop = memref.alloc() : memref<4xf32>
      memref.store %value, %loop[%i] : memref<4xf32>
      func.call @consume_loop(%loop) : (memref<4xf32>) -> ()
    }
    return
  }
}
