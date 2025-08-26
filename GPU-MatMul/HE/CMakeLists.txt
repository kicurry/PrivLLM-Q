cmake_minimum_required(VERSION 3.10)
project(HEBackend LANGUAGES CXX)
set(FATHER_PROJECT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

# Phantom: GPU backend for HE
if(USE_HE_GPU)
    find_package(CUDAToolkit REQUIRED)
    find_package(Phantom REQUIRED PATHS "${FATHER_PROJECT_SOURCE_DIR}/deps/phantom-fhe/build_phantom")
endif()

# Set source files
if(NOT USE_HE_GPU)
    set(srcs 
        src/UnifiedHE.cpp)
else()
    set(srcs 
        src/PhantomWrapper.cpp
        src/UnifiedEvaluator.cpp
        src/UnifiedHE.cpp)
endif()

# Add static library for HE
add_library(HE STATIC ${srcs})

# Add include files for HE
target_include_directories(HE PUBLIC include)

# Link libraries for HE
if(NOT USE_HE_GPU)
    target_link_libraries(HE PUBLIC
        SEAL::seal
        Datatype
    )
else()
    target_compile_definitions(HE PUBLIC USE_HE_GPU)
    target_link_libraries(HE PUBLIC
        SEAL::seal
        phantom::Phantom
        CUDA::cudart # <cuda_runtime_api.h>
        Datatype
    )
endif()
