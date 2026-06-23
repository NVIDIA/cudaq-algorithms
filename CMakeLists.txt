cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

project(cudaq-algorithms LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
include(FetchContent)
include(CMakePackageConfigHelpers)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
include(CUDA-QAlgorithms)

option(CUDAQ_ALGORITHMS_BINDINGS_PYTHON "Build Python bindings." ON)
option(CUDAQ_ALGORITHMS_INCLUDE_TESTS "Build tests." ON)

cudaqa_import_cudaq_targets()

if (SKBUILD)
  # Keep wheel contents under the Python package directory, matching the
  # CUDA-QX QEC/Solvers packaging layout.
  set(CMAKE_INSTALL_BINDIR cudaq_algorithms/bin)
  set(CMAKE_INSTALL_INCLUDEDIR cudaq_algorithms/include)
  set(CMAKE_INSTALL_LIBDIR cudaq_algorithms/lib)
endif()

set(CUDAQ_ALGORITHMS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CUDAQ_ALGORITHMS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CUDAQ_ALGORITHMS_CMAKE_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/CUDA-QAlgorithms"
    CACHE STRING "CMake package install directory.")

if (DEFINED ENV{CUDAQ_ALGORITHMS_VERSION})
  set(CUDAQ_ALGORITHMS_VERSION "$ENV{CUDAQ_ALGORITHMS_VERSION}")
else()
  set(CUDAQ_ALGORITHMS_VERSION "0.0.0")
endif()

execute_process(COMMAND git rev-parse --verify HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE CUDAQ_ALGORITHMS_COMMIT_SHA
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/lib/version.cpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/lib/version.cpp" @ONLY)

add_subdirectory(lib)

if (CUDAQ_ALGORITHMS_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()

if (CUDAQ_ALGORITHMS_INCLUDE_TESTS)
  enable_testing()
  add_subdirectory(lib/unittests)
endif()

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUDA-QAlgorithmsConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/CUDA-QAlgorithmsConfig.cmake"
  INSTALL_DESTINATION ${CUDAQ_ALGORITHMS_CMAKE_INSTALL_DIR}
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/CUDA-QAlgorithmsConfigVersion.cmake"
  VERSION ${CUDAQ_ALGORITHMS_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/CUDA-QAlgorithmsConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/CUDA-QAlgorithmsConfigVersion.cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/CUDA-QAlgorithms.cmake"
  COMPONENT algorithms-cmake
  DESTINATION ${CUDAQ_ALGORITHMS_CMAKE_INSTALL_DIR}
)
