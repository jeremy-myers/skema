cmake_minimum_required(VERSION 3.20)
project(skema)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/include/")
set(INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/tpls/primme/include")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${skema_BINARY_DIR}/bin)

# Target definitions
add_executable(${PROJECT_NAME}
    ${CMAKE_SOURCE_DIR}/driver/main.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_AlgParams.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_DimRedux_Gauss.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_DimRedux_SparseSign.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_EIGSVD.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_Kernel.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_Sampler.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_ISVD_Primme.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_ISVD_MatrixMatvec.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_ISVD.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_SketchySVD.cpp
)

# Add KokkosKernels
find_package(KokkosKernels REQUIRED)
target_link_libraries(${PROJECT_NAME} Kokkos::kokkoskernels)

# Add Lapack
find_package(LAPACKE REQUIRED)
if(LAPACKE_FOUND)
    target_link_libraries(${PROJECT_NAME} ${LAPACKE_LIBRARIES})
    message(STATUS "Lapack libraries: ${LAPACKE_LIBRARIES}")
    get_target_property(${PROJECT_NAME} LAPACK::LAPACK INTERFACE_LINK_LIBRARIES)
    message(STATUS "LAPACK INTERFACE LINK LIBRARIES FROM TARGET: ${targetlib}")
    set(${LAPACK_LINKER_FLAGS} -llapacke -llapack -lblas)
    message(STATUS "Lapack linker flags: ${LAPACKE_LINKER_FLAGS}")
endif()
target_include_directories(${PROJECT_NAME} PRIVATE "/usr/include" "/usr/local/include")

# Add PRIMME
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src/)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include/)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/tpls/primme/include/)
target_link_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/tpls/primme/lib/)
target_link_libraries(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/tpls/primme/lib/libprimme.a -llapack -lblas)

add_compile_options(-fsanitize=address)
add_link_options(-fsanitize=address)

# Googletest
# add_subdirectory(tpls/googletest)
# find_package(GTest CONFIG REQUIRED)

# enable_testing()

# add_executable(
#   test_dim_redux
#   ${CMAKE_SOURCE_DIR}/test/Skema_Test_DimRedux.cpp
# )
# target_link_libraries(
#   test_dim_redux
#   KokkosKernels
#   GTest::gtest_main
# )

# include(GoogleTest)
# gtest_discover_tests(test_dim_redux)