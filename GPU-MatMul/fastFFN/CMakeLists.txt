# Macro function to add executable with common source files
macro(add_ffn_executable target_name main_source)
    add_executable(${target_name}
        ${CMAKE_CURRENT_LIST_DIR}/${main_source}
        common/profile_helper.cpp
        common/memory_monitor.cpp
        common/thread_pool.cpp
        common/utils.cpp
        ffn_utils.cpp
    )
    target_link_libraries(${target_name} PUBLIC HE Datatype)

    if(ENABLE_VALIDATION)
        target_sources(${target_name} PRIVATE common/matrix_validator.cpp)
        target_compile_definitions(${target_name} PRIVATE ENABLE_MATRIX_VALIDATION)
        target_link_libraries(${target_name} PUBLIC Eigen3::Eigen)
    endif()

    # Ensure NDEBUG is defined for Release builds
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_definitions(${target_name} PRIVATE NDEBUG)
    endif()
endmacro()

# Add executables using the macro
add_ffn_executable(test_ffn_baseline ffn_ntt_pmult.cpp)
add_ffn_executable(test_ffn_gsd ffn_less_ntt.cpp)
add_ffn_executable(test_ffn_gsd_nobsgs ffn_less_ntt_nobsgs.cpp)

if(USE_HE_GPU)
    add_ffn_executable(test_ffn_gsd_falr ffn_less_ntt_lazy_reduction.cpp)
endif()
