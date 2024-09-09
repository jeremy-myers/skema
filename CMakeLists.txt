cmake_minimum_required(VERSION 3.20)
project(skema)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/include/")
set(INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/tpls/primme/include")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Target definitions
add_executable(${PROJECT_NAME}
    ${CMAKE_SOURCE_DIR}/driver/main.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_AlgParams.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_EIGSVD.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_Kernel.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_Sampler.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_ISVD_Primme.cpp    
    ${CMAKE_SOURCE_DIR}/src/Skema_ISVD_MatrixMatvec.cpp 
    ${CMAKE_SOURCE_DIR}/src/Skema_ISVD.cpp
    ${CMAKE_SOURCE_DIR}/src/Skema_Primme_MatrixMatvec.cpp
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
# find_package(GTest CONFIG REQUIRED)
# GoogleTest
# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   GIT_REPOSITORY https://github.com/google/googletest.git
#   GIT_TAG v1.14.0
# )
# # For Windows: Prevent overriding the parent project's compiler/linker settings
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest)
# include(GoogleTest)

# set(SKSVD_UNIT_TESTS
#     test/unitTestMain.cpp
#     test/ThreeSketch_unitTest.cpp)
# add_executable(sksvd_unit_test ${SKSVD_UNIT_TESTS})
# target_include_directories(sksvd_unit_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test ${CMAKE_SOURCE_DIR}/src)
# target_link_libraries(sksvd_unit_test GTest::gtest_main Kokkos::kokkoskernels)
