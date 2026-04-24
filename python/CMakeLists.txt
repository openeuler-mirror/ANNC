include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=kpgemm.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(KPGEMMPythonSources)

file(GLOB_RECURSE ALL_PYTHON_SCRIPTS
  ${CMAKE_CURRENT_SOURCE_DIR}/kpgemm *.py
)
set(PYTHON_SCRIPTS "")
foreach(full_path_file ${ALL_PYTHON_SCRIPTS})
  file(RELATIVE_PATH relative_path ${CMAKE_CURRENT_SOURCE_DIR}/kpgemm ${full_path_file})
  list(APPEND PYTHON_SCRIPTS ${relative_path})
endforeach()

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT KPGEMMPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kpgemm"
  TD_FILE dialects/PimpDialect.td
  SOURCES
    ${PYTHON_SCRIPTS}
  DIALECT_NAME pimp)

file(COPY "${CMAKE_SOURCE_DIR}/python/setup.py" 
     DESTINATION "${CMAKE_SOURCE_DIR}/build/python_packages")
file(COPY "${CMAKE_SOURCE_DIR}/python/kpgemm/ops.py"
     DESTINATION "${CMAKE_SOURCE_DIR}/build/python_packages/kpgemm")
file(COPY "${CMAKE_SOURCE_DIR}/python/kpgemm/builder.py"
     DESTINATION "${CMAKE_SOURCE_DIR}/build/python_packages/kpgemm")
file(COPY "${CMAKE_SOURCE_DIR}/python/kpgemm/enums.py"
     DESTINATION "${CMAKE_SOURCE_DIR}/build/python_packages/kpgemm")
file(COPY "${CMAKE_SOURCE_DIR}/python/kpgemm/types.py"
     DESTINATION "${CMAKE_SOURCE_DIR}/build/python_packages/kpgemm")
file(COPY "${CMAKE_SOURCE_DIR}/python/kpgemm/helper.py"
     DESTINATION "${CMAKE_SOURCE_DIR}/build/python_packages/kpgemm")

declare_mlir_python_extension(KPGEMMPythonSources.Extension
  MODULE_NAME _kpgemm
  ADD_TO_PARENT KPGEMMPythonSources
  SOURCES
    PimpModule.cpp
    Attributes.cpp
    Types.cpp

  EMBED_CAPI_LINK_LIBS
    PimpCAPIDialect
    PimpCAPIPasses
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(KPGEMMPythonCAPI
  INSTALL_COMPONENT KPGEMMPythonModules
  INSTALL_DESTINATION python_packages/kpgemm/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/python_packages/kpgemm/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    KPGEMMPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(KPGEMMPythonModules
  ROOT_PREFIX "${CMAKE_SOURCE_DIR}/build/python_packages/kpgemm"
  INSTALL_PREFIX "python_packages/kpgemm"
  DECLARED_SOURCES
    KPGEMMPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    KPGEMMPythonCAPI
  )