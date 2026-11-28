//===----------------------------------------------------------------------===//
// Operation Interpret Interface
//===----------------------------------------------------------------------===//
#ifndef ATIR_INTERPRET_INTFC_TD
#define ATIR_INTERPRET_INTFC_TD
include "mlir/IR/OpBase.td"

def InterpretIntfc : OpInterface<"InterpretIntfc"> {
  let description = [{
  }];
  let cppNamespace = "::atir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          verify generated operation
        }],
        /*retType=*/"void",
        /*methodName=*/"Interpret",
        /*args=*/(ins)
      >,
  ];
}
#endif // ATIR_INTERPRET_INTFC_TD
