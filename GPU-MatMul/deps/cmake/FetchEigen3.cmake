# Use FetchContent to download and build Eigen3
include(FetchContent)

# Prefer using system-installed Eigen3 first
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "System Eigen3 not found, will use local version")

    # Configure Eigen3 download
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0 # Specify version
    )

    # Download and build Eigen3
    FetchContent_MakeAvailable(eigen)

    # Set include directory
    set(Eigen3_INCLUDE_DIR ${eigen_SOURCE_DIR})
    message(STATUS "Using local Eigen3: ${Eigen3_INCLUDE_DIR}")
else()
    message(STATUS "Using system Eigen3: ${EIGEN3_INCLUDE_DIRS}")
endif()
