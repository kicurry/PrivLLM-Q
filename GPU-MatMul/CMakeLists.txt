cmake_minimum_required(VERSION 3.10)

project(HE_Quant)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_minimum_required(VERSION 3.10)

# Set default build type to Release
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")

option(USE_HE_GPU "Use GPU backend for HE" ON)

# Add validation option
option(ENABLE_VALIDATION "Enable matrix multiplication validation" OFF)

if(ENABLE_VALIDATION)
    # Install Eigen3 using FetchContent method
    include(${CMAKE_SOURCE_DIR}/deps/cmake/FetchEigen3.cmake)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Print current configuration
message(STATUS "==== Current configuration ====")
message(STATUS "CMAKE_BUILD_TYPE :  ${CMAKE_BUILD_TYPE}")
message(STATUS "USE_HE_GPU       :  ${USE_HE_GPU}")
message(STATUS "ENABLE_VALIDATION:  ${ENABLE_VALIDATION}")
message(STATUS "===============================")

add_subdirectory(deps/SEAL)
add_subdirectory(Datatype)
add_subdirectory(HE)
add_subdirectory(fastFFN)
