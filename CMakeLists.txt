cmake_minimum_required(VERSION 3.20)

#---- Declaring the project name will define ${Skema_SOURCE_DIR} to be
#---- the path to this source file, and ${Skema_BINARY_DIR} to be the path
#---- to the directory where cmake has been invoked.
project(Skema VERSION "1.0.0")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Skema_BINARY_DIR}/bin)

include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 52eb8108c5bdec04579160ae17225d66034bd723
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

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

# Nlohmann json
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(tpls/json)

include_directories(
    ${Kokkos_INCLUDE_DIRS_RET}
    ${KokkosKernels_INCLUDE_DIRS_RET}
    ${PRIMME_INCLUDE_DIR}
    ${LAPACK_INCLUDE_DIR}
    ${Skema_SOURCE_DIR}/src/
    ${Skema_SOURCE_DIR}/driver/
    ${Skema_SOURCE_DIR}/tpls/json/single_include/nlohmann/
)

# -------------
# Skema Library
# -------------

# List source files to be built
set(Skema_SOURCE_LIST
    ${Skema_SOURCE_DIR}/src/Skema_AlgParams.cpp
    ${Skema_SOURCE_DIR}/src/Skema_IO.cpp
    ${Skema_SOURCE_DIR}/src/Skema_EIGSVD.cpp
    ${Skema_SOURCE_DIR}/src/Skema_EIGSVD_MatrixMatvec.cpp
    ${Skema_SOURCE_DIR}/src/Skema_Kernel.cpp
    ${Skema_SOURCE_DIR}/src/Skema_Sampler.cpp
    ${Skema_SOURCE_DIR}/src/Skema_ISVD_Primme.cpp
    ${Skema_SOURCE_DIR}/src/Skema_ISVD_MatrixMatvec.cpp
    ${Skema_SOURCE_DIR}/src/Skema_ISVD.cpp
    ${Skema_SOURCE_DIR}/src/Skema_DimRedux_Gauss.cpp
    ${Skema_SOURCE_DIR}/src/Skema_DimRedux_SparseSign.cpp
    ${Skema_SOURCE_DIR}/src/Skema_Sketchy_SPD.cpp
    ${Skema_SOURCE_DIR}/src/Skema_Sketchy_SVD.cpp
)

# Future: if add any special libraries or functionality through if/else or macros,
# append to Skema_SOURCE_LIST as: 
#       set(Skema_SOURCE_LIST ${Skema_SOURCE_LIST} <new source file>)

# Build the library
add_library(skemalib ${Skema_SOURCE_LIST})

target_link_libraries(skemalib PUBLIC ${LAPACK_LIBRARIES})
if(LAPACK_FOUND)
    target_compile_definitions(skemalib PRIVATE LAPACK_FOUND=${LAPACK_FOUND})
endif()

target_link_libraries(skemalib PUBLIC Kokkos::kokkos)
target_link_libraries(skemalib PUBLIC Kokkos::kokkoskernels)
target_link_libraries(skemalib PUBLIC primme)
target_link_libraries(skemalib PRIVATE nlohmann_json::nlohmann_json)

install(TARGETS skemalib
        EXPORT skema
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
file(GLOB skema_headers ${Skema_SOURCE_DIR}/*.hpp)
install(FILES ${skema_headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/skema
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
        )

# Target definitions
add_executable(skema
    ${CMAKE_SOURCE_DIR}/driver/main.cpp
)
target_link_libraries(skema skemalib)
install(TARGETS skema
        EXPORT skema
        DESTINATION ${CMAKE_INSTALL_BINDIR}
)

add_executable(convert
    ${CMAKE_SOURCE_DIR}/driver/convert.cpp
)
target_link_libraries(convert skemalib)

# Testing
# add_executable(example example.cpp)
# target_link_libraries(example gtest_main)
# add_test(NAME example_test COMMAND example)
include_directories(${Skema_SOURCE_DIR}/test)

set(Skema_SOURCE_TEST
    ${Skema_SOURCE_DIR}/test/Unit_Tests_Main.cpp
    ${Skema_SOURCE_DIR}/test/Skema_Test_Kernel.cpp
)

add_executable(skema_unit_tests ${Skema_SOURCE_TEST})
target_link_libraries(skema_unit_tests gtest skemalib)