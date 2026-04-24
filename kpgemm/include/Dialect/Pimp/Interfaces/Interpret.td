//===----------------------------------------------------------------------===//
// Operation Interpret Interface
//===----------------------------------------------------------------------===//
#ifndef PIMP_INTERPRET_INTFC_TD
#define PIMP_INTERPRET_INTFC_TD
include "mlir/IR/OpBase.td"

def InterpretIntfc : OpInterface<"InterpretIntfc"> {
  let description = [{
  }];
  let cppNamespace = "::pimp";
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
#endif // PIMP_INTERPRET_INTFC_TD
