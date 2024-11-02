cmake_minimum_required(VERSION 3.20)

#---- Declaring the project name will define ${Skema_SOURCE_DIR} to be
#---- the path to this source file, and ${Skema_BINARY_DIR} to be the path
#---- to the directory where cmake has been invoked.
project(Skema VERSION "1.0.0")

# Add primme build with specific commit to eloy/index-conv-test-fun

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

if(Kokkos_ENABLE_OPENMP)
    set(HAVE_OPENMP ON)
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

include_directories(
    ${Kokkos_INCLUDE_DIRS_RET}
    ${KokkosKernels_INCLUDE_DIRS_RET}
    ${Skema_SOURCE_DIR}/src/
    ${Skema_SOURCE_DIR}/driver/
    ${Skema_SOURCE_DIR}/tpls/primme/include/
    ${LAPACK_INCLUDE_DIR}
)

# Add PRIMME
# add_library(primme STATIC
#     ${Skema_SOURCE_DIR}/tpls/primme/src/eigs/auxiliary_eigs_normal.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/eigs/auxiliary_eigs.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/eigs/convergence.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/eigs/correction.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/eigs/factorize.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/eigs/init.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/eigs/inner_solve.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/eigs/main_iter.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/eigs/ortho.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/eigs/primme_c.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/eigs/primme_f77.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/eigs/primme_interface.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/eigs/restart.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/eigs/solve_projection.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/eigs/update_projection.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/eigs/update_W.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/linalg/auxiliary.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/linalg/blaslapack.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/linalg/cublas_wrapper.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/linalg/magma_wrapper.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/linalg/memman.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/linalg/wtime.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/svds/primme_svds_c.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/svds/primme_svds_f77.c
#     ${Skema_SOURCE_DIR}/tpls/primme/src/svds/primme_svds_interface.c
# )

# find_package(LAPACK REQUIRED)
# target_link_libraries(
#     primme
#     LAPACK::LAPACK -llapack -lblas
# )
# target_include_directories(
#     primme PRIVATE
#     ${Skema_SOURCE_DIR}/tpls/primme/include
#     ${Skema_SOURCE_DIR}/tpls/primme/src/eigs
#     ${Skema_SOURCE_DIR}/tpls/primme/src/include
#     ${Skema_SOURCE_DIR}/tpls/primme/src/linalg
#     ${Skema_SOURCE_DIR}/tpls/primme/src/svds
# )
# install(
#     TARGETS primme
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/lib
#     EXPORT primme
# )



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

target_link_libraries(${PROJECT_NAME} ${LAPACK_LIBRARIES} -llapack -lblas)
target_link_libraries(${PROJECT_NAME} Kokkos::kokkos)
target_link_libraries(${PROJECT_NAME} Kokkos::kokkoskernels)
target_link_libraries(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/tpls/primme/lib/libprimme.a -llapack -lblas)

if(LAPACK_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE LAPACK_FOUND=${LAPACK_FOUND})
endif()

add_compile_options(-fsanitize=address)
add_link_options(-fsanitize=address)
