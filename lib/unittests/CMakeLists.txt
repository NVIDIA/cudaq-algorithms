# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(GTest CONFIG QUIET)

if (NOT GTest_FOUND)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(googletest)
endif()

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

if (CMAKE_COMPILER_IS_GNUCXX
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
  target_compile_options(gtest PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:--param=evrp-mode=legacy>)
endif()

include(GoogleTest)

add_custom_target(CUDAQAlgorithmsUnitTests)

function(cudaqa_add_unittest test_name)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name}
    PRIVATE
      GTest::gtest_main
      cudaq-algorithms
      cudaq::cudaq
      cudaq::cudaq-${CUDAQ_TARGET}-target
  )
  target_link_options(${test_name} PRIVATE LINKER:--no-as-needed)
  add_dependencies(CUDAQAlgorithmsUnitTests ${test_name})
  gtest_discover_tests(${test_name})
endfunction()

cudaqa_add_unittest(test_ceo_operator_pool)
cudaqa_add_unittest(test_uccgsd_operator_pool)
cudaqa_add_unittest(test_operator_pools)
cudaqa_add_unittest(test_upccgsd_operator_pool)
cudaqa_add_unittest(test_uccsd)
cudaqa_add_unittest(test_bravyi_kitaev)
cudaqa_add_unittest(test_block_encoding)
cudaqa_add_unittest(test_qubitization)
cudaqa_add_unittest(test_qsvt)
cudaqa_add_unittest(test_krylov)
