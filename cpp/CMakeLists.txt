cmake_minimum_required(VERSION 3.18)
project(step2point LANGUAGES CXX)

option(STEP2POINT_BUILD_PYTHON "Build Python bindings" OFF)
option(STEP2POINT_BUILD_TESTS "Build C++ tests" ON)

add_library(step2point_core
    src/shower.cpp
    src/identity.cpp
    src/merge_within_cell.cpp
)

target_include_directories(step2point_core PUBLIC include)
target_compile_features(step2point_core PUBLIC cxx_std_17)

if(STEP2POINT_BUILD_TESTS)
    enable_testing()
    add_executable(test_merge_within_cell tests/test_merge_within_cell.cpp)
    target_link_libraries(test_merge_within_cell PRIVATE step2point_core)
    add_test(NAME test_merge_within_cell COMMAND test_merge_within_cell)
endif()

if(STEP2POINT_BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(_step2point_cpp bindings/pybind_module.cpp)
    target_link_libraries(_step2point_cpp PRIVATE step2point_core)
endif()
