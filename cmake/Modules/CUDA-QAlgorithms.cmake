# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

function(cudaqa_add_device_code LIBRARY_NAME)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs SOURCES COMPILER_FLAGS DEPENDS_ON)
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT DEFINED CUDAQ_INSTALL_DIR)
    message(FATAL_ERROR "CUDAQ_INSTALL_DIR must be defined")
  endif()

  if(NOT ARGS_SOURCES)
    message(FATAL_ERROR "At least one SOURCE file is required")
  endif()

  set(COMPILER ${CUDAQ_INSTALL_DIR}/bin/nvq++)

  if (CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN)
    set(ARGS_COMPILER_FLAGS "${ARGS_COMPILER_FLAGS} --gcc-install-dir=${CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN}")
  endif()

  set(prop "$<TARGET_PROPERTY:${LIBRARY_NAME},INCLUDE_DIRECTORIES>")
  foreach(source ${ARGS_SOURCES})
    get_filename_component(filename ${source} NAME_WE)
    set(output_file "${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}_${filename}.o")
    cmake_path(GET output_file FILENAME baseName)

    add_custom_command(
      OUTPUT ${output_file}
      COMMAND ${COMPILER}
        ${ARGS_COMPILER_FLAGS} -c -fPIC
        ${CMAKE_CURRENT_SOURCE_DIR}/${source} -o ${baseName}
        "$<$<BOOL:${prop}>:-I $<JOIN:${prop}, -I >>"
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source} ${ARGS_DEPENDS_ON}
      COMMENT "Compiling ${source} with nvq++"
      VERBATIM
    )

    list(APPEND object_files ${output_file})
    list(APPEND custom_targets ${LIBRARY_NAME}_${filename}_target)
    add_custom_target(${LIBRARY_NAME}_${filename}_target DEPENDS ${output_file})
  endforeach()

  add_dependencies(${LIBRARY_NAME} ${custom_targets})
  target_sources(${LIBRARY_NAME} PRIVATE ${object_files})
endfunction()

function(_cudaqa_import_nvqir_target target_name library_name)
  if(NOT TARGET ${target_name})
    add_library(${target_name} SHARED IMPORTED)
    set_target_properties(${target_name} PROPERTIES
      IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/${library_name}${CMAKE_SHARED_LIBRARY_SUFFIX}"
      IMPORTED_SONAME "${library_name}${CMAKE_SHARED_LIBRARY_SUFFIX}"
      IMPORTED_LINK_INTERFACE_LIBRARIES "cudaq::cudaq-platform-default;cudaq::cudaq-em-default")
  endif()
endfunction()

function(cudaqa_import_cudaq_targets)
  if (NOT CUDAQ_DIR)
    message(FATAL_ERROR
      "CUDAQ_DIR must point to the CUDA-Q CMake package directory, e.g. "
      "<cudaq-prefix>/lib/cmake/cudaq.")
  endif()

  set(CUDAQ_CMAKE_DIR "${CUDAQ_DIR}")
  get_filename_component(_cudaq_parent_dir "${CUDAQ_CMAKE_DIR}" DIRECTORY)
  get_filename_component(CUDAQ_LIBRARY_DIR "${_cudaq_parent_dir}" DIRECTORY)
  get_filename_component(CUDAQ_INSTALL_DIR "${CUDAQ_LIBRARY_DIR}" DIRECTORY)
  set(CUDAQ_INCLUDE_DIR "${CUDAQ_INSTALL_DIR}/include")

  include(CMakeFindDependencyMacro)
  set(NVQIR_DIR "${_cudaq_parent_dir}/nvqir")

  foreach(_cudaq_pkg IN ITEMS
      CUDAQCommon
      CUDAQEmDefault
      CUDAQEnsmallen
      CUDAQLogger
      CUDAQMlirRuntime
      CUDAQNlopt
      CUDAQOperator
      CUDAQPlatformDefault
      CUDAQPythonInterop)
    set(${_cudaq_pkg}_DIR "${CUDAQ_CMAKE_DIR}")
  endforeach()

  find_package(NVQIR REQUIRED CONFIG)
  find_package(CUDAQOperator REQUIRED CONFIG)
  find_package(CUDAQCommon REQUIRED CONFIG)
  find_package(CUDAQNlopt REQUIRED CONFIG)
  find_package(CUDAQEnsmallen REQUIRED CONFIG)
  find_package(CUDAQEmDefault REQUIRED CONFIG)
  find_package(CUDAQPlatformDefault REQUIRED CONFIG)
  find_package(CUDAQPythonInterop CONFIG)

  if(NOT TARGET cudaq::cudaq)
    include("${CUDAQ_CMAKE_DIR}/CUDAQTargets.cmake")
  endif()

  set(__base_nvtarget_name "custatevec")
  find_library(CUDAQ_CUSVSIM_PATH NAMES cusvsim-fp32 HINTS ${CUDAQ_LIBRARY_DIR})
  if (CUDAQ_CUSVSIM_PATH)
    set(__base_nvtarget_name "cusvsim")
  endif()

  _cudaqa_import_nvqir_target(cudaq::cudaq-default-target "libnvqir-${__base_nvtarget_name}-fp64")
  _cudaqa_import_nvqir_target(cudaq::cudaq-nvidia-target "libnvqir-${__base_nvtarget_name}-fp32")
  _cudaqa_import_nvqir_target(cudaq::cudaq-nvidia-fp64-target "libnvqir-${__base_nvtarget_name}-fp64")
  _cudaqa_import_nvqir_target(cudaq::cudaq-nvidia-mgpu-target "libnvqir-mgpu-fp32")
  _cudaqa_import_nvqir_target(cudaq::cudaq-nvidia-mgpu-fp64-target "libnvqir-mgpu-fp64")
  _cudaqa_import_nvqir_target(cudaq::cudaq-qpp-cpu-target "libnvqir-qpp")
  _cudaqa_import_nvqir_target(cudaq::cudaq-qpp-density-matrix-cpu-target "libnvqir-dm")
  _cudaqa_import_nvqir_target(cudaq::cudaq-stim-target "libnvqir-stim")

  if(NOT COMMAND cudaq_set_target)
    function(cudaq_set_target TARGETNAME)
      message(STATUS "CUDA Quantum Target = ${TARGETNAME}")
      target_link_libraries(cudaq::cudaq INTERFACE cudaq::cudaq-${TARGETNAME}-target)
    endfunction()
  endif()

  set(CUDAQ_TARGET "qpp-cpu" CACHE STRING
      "The CUDA Quantum target to compile for and execute on.")
  cudaq_set_target(${CUDAQ_TARGET})

  foreach(_cudaq_var IN ITEMS
      CUDAQ_CMAKE_DIR
      CUDAQ_INCLUDE_DIR
      CUDAQ_INSTALL_DIR
      CUDAQ_LIBRARY_DIR
      CUDAQPythonInterop_DIR)
    set(${_cudaq_var} "${${_cudaq_var}}" PARENT_SCOPE)
  endforeach()
endfunction()
