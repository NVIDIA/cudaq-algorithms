find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

if(NOT COMMAND nanobind_add_module)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(NANOBIND_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
  endif()
  find_package(nanobind CONFIG QUIET)
endif()

if(NOT COMMAND nanobind_add_module)
  FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind
    GIT_TAG v2.12.0
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(nanobind)
endif()

set(MODULE_NAME _pycudaq_algorithms)

nanobind_add_module(${MODULE_NAME}
  bindings/cudaq_algorithms.cpp
  bindings/py_fermion.cpp
  bindings/py_stateprep.cpp
  bindings/py_block_encoding.cpp
)

target_include_directories(${MODULE_NAME} PRIVATE .)

target_link_libraries(${MODULE_NAME}
  PRIVATE
    cudaq-algorithms
    cudaq::cudaq
    cudaq::cudaq-python-interop
)

set_target_properties(${MODULE_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cudaq_algorithms")

if (NOT SKBUILD)
  set_target_properties(${MODULE_NAME} PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN/../lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE)
else()
  set_target_properties(${MODULE_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN/lib:$ORIGIN/../lib")
endif()

add_custom_target(copy_cudaq_algorithms_python_files ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/cudaq_algorithms
          ${CMAKE_BINARY_DIR}/python/cudaq_algorithms
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cudaq_algorithms
)

add_dependencies(${MODULE_NAME} copy_cudaq_algorithms_python_files)

install(DIRECTORY cudaq_algorithms
  COMPONENT algorithms-python
  DESTINATION .)

install(TARGETS ${MODULE_NAME}
  COMPONENT algorithms-python
  DESTINATION cudaq_algorithms/
)
