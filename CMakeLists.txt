cmake_minimum_required(VERSION 3.20)

#---- Declaring the project name will define ${Skema_SOURCE_DIR} to be
#---- the path to this source file, and ${Skema_BINARY_DIR} to be the path
#---- to the directory where cmake has been invoked.
project(Skema VERSION "1.0.0")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Skema_BINARY_DIR}/bin)

find_package(LAPACK REQUIRED)
if(LAPACK_FOUND)
    set(LAPACK_FOUND TRUE)
endif()

# Kokkos build options:
#  - Inline build from tpls/kokkos
#  - Link to external library from Kokkos_ROOT.
if(Kokkos_ROOT)
    set(KOKKOS_INLINE_BUILD OFF)
else()
    set(KOKKOS_INLINE_BUILD ON)
endif()
message(STATUS "Inline Kokkos build is ${KOKKOS_INLINE_BUILD}")

# KokkosKernels build options:
#  - Inline build from tpls/kokkos-kernels
#  - Link to external library from KokkosKernels_ROOT.
if(KokkosKernels_ROOT)
    set(KOKKOSKERNELS_INLINE_BUILD OFF)
else()
    set(KOKKOSKERNELS_INLINE_BUILD ON)
endif()
message(STATUS "Inline Kokkos-Kernels build is ${KOKKOSKERNELS_INLINE_BUILD}")

if(PRIMME_ROOT)
    set(PRIMME_INLINE_BUILD OFF)
else()
    set(PRIMME_INLINE_BUILD ON)
endif()
message(STATUS "Inline PRIMME build is ${PRIMME_INLINE_BUILD}")

if(Kokkos_ENABLE_OPENMP)
    set(HAVE_OPENMP ON)
    set(PRIMME_ENABLE_OPENMP ON)
endif()

if(KOKKOS_INLINE_BUILD)
    # Add Kokkos
    add_subdirectory(tpls/kokkos)
    set(HAVE_KOKKOS ON)
else()
    # Link against installed Kokkos
    find_package(Kokkos REQUIRED)
    message("-- Skema:  Using Kokkos:  ${Kokkos_CONFIG}")
    set(HAVE_KOKKOS ON)
endif()

if(KOKKOSKERNELS_INLINE_BUILD)
    # Add KokkosKernels
    add_subdirectory(tpls/kokkos-kernels)
    set(HAVE_KOKKOS_KERNELS ON)
else()
    find_package(KokkosKernels REQUIRED)
    message("-- Skema:  Using KokkosKernels:  ${KokkosKernels_CONFIG}")
    set(HAVE_KOKKOS_KERNELS ON)
endif()

if(PRIMME_INLINE_BUILD)
    # Add PRIMME
    add_subdirectory(tpls/primme)
    set(HAVE_PRIMME ON)
else()
    # Link against installed PRIMME
    find_package(primme REQUIRED)
    message("--- Skema: Using PRIMME: ${primme_CONFIG}")
    set(HAVE_PRIMME ON)
endif()

# Nlohmann jons
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(tpls/json)

include_directories(
    ${Skema_SOURCE_DIR}/src/
    ${Skema_SOURCE_DIR}/driver/
    ${Kokkos_INCLUDE_DIRS_RET}
    ${KokkosKernels_INCLUDE_DIRS_RET}
    ${PRIMME_INCLUDE_DIR}
    ${LAPACK_INCLUDE_DIR}
    ${Skema_SOURCE_DIR}/tpls/json/single_include/nlohmann/
)

# Target definitions
add_executable(${PROJECT_NAME}
    ${CMAKE_SOURCE_DIR}/driver/main.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_AlgParams.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_EIGSVD.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_EIGSVD_MatrixMatvec.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_Kernel.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_Sampler.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_ISVD_Primme.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_ISVD_MatrixMatvec.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_ISVD.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_DimRedux_Gauss.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_DimRedux_SparseSign.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_SketchySVD.cpp
)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "skema")

target_link_libraries(${PROJECT_NAME} PUBLIC ${LAPACK_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC Kokkos::kokkos)
target_link_libraries(${PROJECT_NAME} PUBLIC Kokkos::kokkoskernels)
target_link_libraries(${PROJECT_NAME} PUBLIC primme)
target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann_json::nlohmann_json)

if(LAPACK_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE LAPACK_FOUND=${LAPACK_FOUND})
endif()


# Target definitions
add_executable(convert
    ${CMAKE_SOURCE_DIR}/driver/convert.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_AlgParams.cpp
)
set_target_properties(convert PROPERTIES OUTPUT_NAME "convert")

target_link_libraries(convert PUBLIC ${LAPACK_LIBRARIES})
target_link_libraries(convert PUBLIC Kokkos::kokkos)
target_link_libraries(convert PUBLIC Kokkos::kokkoskernels)
target_link_libraries(convert PUBLIC primme)

if(LAPACK_FOUND)
    target_compile_definitions(convert PRIVATE LAPACK_FOUND=${LAPACK_FOUND})
endif()
