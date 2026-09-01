include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=annc.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(ANNCPythonSources)

file(GLOB_RECURSE ALL_PYTHON_SCRIPTS
  ${CMAKE_CURRENT_SOURCE_DIR}/annc *.py
)
set(PYTHON_SCRIPTS "")
foreach(full_path_file ${ALL_PYTHON_SCRIPTS})
  file(RELATIVE_PATH relative_path ${CMAKE_CURRENT_SOURCE_DIR}/annc ${full_path_file})
  list(APPEND PYTHON_SCRIPTS ${relative_path})
endforeach()

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT ANNCPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/annc"
  TD_FILE dialects/AtirDialect.td
  SOURCES
    ${PYTHON_SCRIPTS}
  DIALECT_NAME atir)

file(COPY "${CMAKE_SOURCE_DIR}/python/setup.py" 
     DESTINATION "${CMAKE_SOURCE_DIR}/build/python_packages")
file(COPY "${CMAKE_SOURCE_DIR}/python/annc/ops.py"
     DESTINATION "${CMAKE_SOURCE_DIR}/build/python_packages/annc")
file(COPY "${CMAKE_SOURCE_DIR}/python/annc/builder.py"
     DESTINATION "${CMAKE_SOURCE_DIR}/build/python_packages/annc")
file(COPY "${CMAKE_SOURCE_DIR}/python/annc/enums.py"
     DESTINATION "${CMAKE_SOURCE_DIR}/build/python_packages/annc")
file(COPY "${CMAKE_SOURCE_DIR}/python/annc/types.py"
     DESTINATION "${CMAKE_SOURCE_DIR}/build/python_packages/annc")
file(COPY "${CMAKE_SOURCE_DIR}/python/annc/helper.py"
     DESTINATION "${CMAKE_SOURCE_DIR}/build/python_packages/annc")

declare_mlir_python_extension(ANNCPythonSources.Extension
  MODULE_NAME _annc
  ADD_TO_PARENT ANNCPythonSources
  SOURCES
    AtirModule.cpp
    Attributes.cpp
    Types.cpp

  EMBED_CAPI_LINK_LIBS
    AtirCAPIDialect
    AtirCAPIPasses
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(ANNCPythonCAPI
  INSTALL_COMPONENT ANNCPythonModules
  INSTALL_DESTINATION python_packages/annc/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/python_packages/annc/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    ANNCPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

# Force AtirPasses into the common CAPI shared library with whole-archive
target_link_libraries(ANNCPythonCAPI PRIVATE "-Wl,--whole-archive" AtirPasses "-Wl,--no-whole-archive")

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(ANNCPythonModules
  ROOT_PREFIX "${CMAKE_SOURCE_DIR}/build/python_packages/annc"
  INSTALL_PREFIX "python_packages/annc"
  DECLARED_SOURCES
    ANNCPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    ANNCPythonCAPI
  )

# The ANNC python modules targets symlink generated MLIR Python files (like
# _ops_gen.py). When building with -j, the ANNC sub-targets can race ahead
# of the MLIR targets that generate these files. Add per-target dependencies
# to ensure each ANNC symlink target waits for its MLIR counterpart.
_flatten_mlir_python_targets(_all_mlir_src_targets MLIRPythonSources)
foreach(_src_tgt ${_all_mlir_src_targets})
  get_target_property(_src_type ${_src_tgt} mlir_python_SOURCES_TYPE)
  if(_src_type STREQUAL "pure")
    set(_annc_tgt "ANNCPythonModules.sources.${_src_tgt}")
    set(_mlir_mod_tgt "MLIRPythonModules.sources.${_src_tgt}")
    if(TARGET ${_annc_tgt} AND TARGET ${_mlir_mod_tgt})
      add_dependencies(${_annc_tgt} ${_mlir_mod_tgt})
    endif()
  endif()
endforeach()