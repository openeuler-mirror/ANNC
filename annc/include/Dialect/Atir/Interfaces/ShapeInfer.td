//===----------------------------------------------------------------------===//
// Shape Inference Interface
//===----------------------------------------------------------------------===//
#ifndef ATIR_SHAPE_INTFC_TD
#define ATIR_SHAPE_INTFC_TD
include "mlir/IR/OpBase.td"

def ShapeInferIntfc : OpInterface<"ShapeInferIntfc"> {
  let description = [{
  }];
  let cppNamespace = "::atir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          if output shape is dynamic, inference output shape by input shape;
          or verify output shape
        }],
        /*retType=*/"void",
        /*methodName=*/"inferShape",
        /*args=*/(ins)
      >,
  ];
}
#endif // ATIR_SHAPE_INTFC_TD