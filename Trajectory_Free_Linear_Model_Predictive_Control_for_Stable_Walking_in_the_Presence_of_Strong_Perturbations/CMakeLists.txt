cmake_minimum_required(VERSION 3.10)
project(example)
include(FetchContent)

function(enable_ninja_output_coloring target)
    if(UNIX AND CMAKE_GENERATOR STREQUAL "Ninja")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            target_compile_options(${target} PRIVATE -fcolor-diagnostics)
            target_compile_options(${target} PRIVATE -fcolor-diagnostics)
        endif()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(${target} PRIVATE -fdiagnostics-color)
            target_compile_options(${target} PRIVATE -fdiagnostics-color)
        endif()
    endif()
endfunction(enable_ninja_output_coloring)

function(compile_target target)
    enable_ninja_output_coloring(${target})
    target_include_directories(${target} PUBLIC Eigen3::Eigen OsqpEigen::OsqpEigen ${osqp_INCLUDE_DIR})
    target_link_libraries(${target} Eigen3::Eigen OsqpEigen::OsqpEigen)
endfunction(compile_target)

FetchContent_Declare(
    gnuplot
    GIT_REPOSITORY https://github.com/martinruenz/gnuplot-cpp.git
    GIT_TAG origin/master)
FetchContent_MakeAvailable(gnuplot)

include_directories(${gnuplot_SOURCE_DIR}/include)
message("${gnuplot_SOURCE_DIR}")
message("${gnuplot_BINARY_DIR}")

set(EIGEN_NO_DEBUG ON)
find_package(Eigen3 REQUIRED)
find_package(OsqpEigen REQUIRED)

add_executable(example generate_trajectory.cpp)
compile_target(example)
