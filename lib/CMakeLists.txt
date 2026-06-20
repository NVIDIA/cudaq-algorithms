add_library(cudaq-algorithms SHARED
  fermion/jordan_wigner.cpp
  fermion/bravyi_kitaev.cpp
  stateprep/excitations.cpp
  stateprep/fixed_parameter_ucc.cpp
  stateprep/hartree_fock.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
)

target_include_directories(cudaq-algorithms
  PUBLIC
    $<BUILD_INTERFACE:${CUDAQ_ALGORITHMS_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${CUDAQ_INCLUDE_DIR}
)

target_link_libraries(cudaq-algorithms
  PUBLIC
    cudaq::cudaq-operator
    cudaq::cudaq
  PRIVATE
    cudaq::cudaq-common
)

set_target_properties(cudaq-algorithms PROPERTIES
  EXPORT_NAME algorithms
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if (NOT SKBUILD)
  set_target_properties(cudaq-algorithms PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE)
else()
  set_target_properties(cudaq-algorithms PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:$ORIGIN/../../lib")
endif()

add_subdirectory(stateprep/device)

install(TARGETS cudaq-algorithms
  EXPORT CUDA-QAlgorithmsTargets
  COMPONENT algorithms-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT CUDA-QAlgorithmsTargets
  FILE CUDA-QAlgorithmsTargets.cmake
  NAMESPACE cudaq::
  COMPONENT algorithms-cmake
  DESTINATION ${CUDAQ_ALGORITHMS_CMAKE_INSTALL_DIR}
)

install(DIRECTORY ${CUDAQ_ALGORITHMS_INCLUDE_DIR}/cudaq
  COMPONENT algorithms-headers
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h"
)
