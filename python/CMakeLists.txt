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
# The distro MLIR is installed without MLIR_INSTALL_AGGREGATE_OBJECTS, so the
# stock add_mlir_python_common_capi_library() cannot rebuild an aggregate from
# the imported MLIR CAPI targets. llvm-toolset-20-python3-mlir ships
# libMLIRPythonCAPI.so, a self-contained MLIR C API that composes correctly with
# libMLIR.so (the distro's own mlir Python package uses it). Link that and embed
# only ANNC's own C API objects.
add_mlir_aggregate(ANNCPythonCAPI
  SHARED
  DISABLE_INSTALL
  EMBED_LIBS
    AtirCAPIDialect
    AtirCAPIPasses
  PUBLIC_LIBS
    ${ANNC_MLIR_PYTHON_CAPI_LIBRARY}
)
set_target_properties(ANNCPythonCAPI PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/python_packages/annc/_mlir_libs"
  BINARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/python_packages/annc/_mlir_libs"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/python_packages/annc/_mlir_libs"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/python_packages/annc/_mlir_libs"
)
mlir_python_setup_extension_rpath(ANNCPythonCAPI
  RELATIVE_INSTALL_ROOT "../../../..")
install(TARGETS ANNCPythonCAPI
  COMPONENT ANNCPythonModules
  LIBRARY DESTINATION python_packages/annc/_mlir_libs
  RUNTIME DESTINATION python_packages/annc/_mlir_libs
)

# Force AtirPasses into the common CAPI shared library with whole-archive
target_link_libraries(ANNCPythonCAPI PRIVATE "-Wl,--whole-archive" AtirPasses "-Wl,--no-whole-archive")

# Every Python extension is linked with an $ORIGIN RPATH. Stage the distro's
# self-contained MLIR Python C API library next to them so the modules resolve
# libMLIRPythonCAPI.so.20.1 without relying on the host's loader path.
get_filename_component(_annc_mlir_py_capi_real "${ANNC_MLIR_PYTHON_CAPI_LIBRARY}" REALPATH)
get_filename_component(_annc_mlir_py_capi_name "${_annc_mlir_py_capi_real}" NAME)
set(_annc_python_libs_dir "${CMAKE_SOURCE_DIR}/build/python_packages/annc/_mlir_libs")
file(MAKE_DIRECTORY "${_annc_python_libs_dir}")
file(COPY "${_annc_mlir_py_capi_real}" DESTINATION "${_annc_python_libs_dir}")
install(FILES "${_annc_mlir_py_capi_real}"
        DESTINATION python_packages/annc/_mlir_libs)

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
    AArch64Codegen
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