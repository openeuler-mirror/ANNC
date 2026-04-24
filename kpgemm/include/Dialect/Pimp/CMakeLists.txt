set(LLVM_TARGET_DEFINITIONS PimpOps.td)
mlir_tablegen(PimpOps.h.inc -gen-op-decls)
mlir_tablegen(PimpOps.cpp.inc -gen-op-defs)
mlir_tablegen(PimpEnums.h.inc -gen-enum-decls)
mlir_tablegen(PimpEnums.cpp.inc -gen-enum-defs)
add_mlir_doc(PimpOps PimpOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS PimpDialect.td)
mlir_tablegen(PimpDialect.h.inc -gen-dialect-decls)
mlir_tablegen(PimpDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(PimpAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(PimpAttr.cpp.inc -gen-attrdef-defs)
add_mlir_doc(PimpDialect PimpDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS PimpTypes.td)
mlir_tablegen(PimpTypes.h.inc -gen-typedef-decls)
mlir_tablegen(PimpTypes.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(PimpTableGen)

add_subdirectory(Interfaces)
add_subdirectory(Passes)
add_subdirectory(OpVerify)