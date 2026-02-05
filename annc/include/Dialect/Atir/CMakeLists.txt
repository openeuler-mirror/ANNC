set(LLVM_TARGET_DEFINITIONS AtirOps.td)
mlir_tablegen(AtirOps.h.inc -gen-op-decls)
mlir_tablegen(AtirOps.cpp.inc -gen-op-defs)
mlir_tablegen(AtirEnums.h.inc -gen-enum-decls)
mlir_tablegen(AtirEnums.cpp.inc -gen-enum-defs)
add_mlir_doc(AtirOps AtirOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS AtirDialect.td)
mlir_tablegen(AtirDialect.h.inc -gen-dialect-decls)
mlir_tablegen(AtirDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(AtirAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(AtirAttr.cpp.inc -gen-attrdef-defs)
add_mlir_doc(AtirDialect AtirDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS AtirTypes.td)
mlir_tablegen(AtirTypes.h.inc -gen-typedef-decls)
mlir_tablegen(AtirTypes.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(AtirTableGen)

add_subdirectory(Interfaces)
add_subdirectory(Passes)
add_subdirectory(OpVerify)