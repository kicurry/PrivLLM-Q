cmake_minimum_required(VERSION 3.10)
project(Datatype_test LANGUAGES CXX)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(Datatype STATIC ${srcs})
target_include_directories(Datatype PUBLIC include)

message(STATUS "Datatype_FOUND: ${Datatype_FOUND}")