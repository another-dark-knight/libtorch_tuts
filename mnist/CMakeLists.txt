cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

set(CMAKE_CXX_FLAGS "-std=c++14 -Ofast -march=native")
set(CMAKE_PREFIX_PATH "/home/spandan/pytorch")
find_package(Torch REQUIRED)
#find_package(ATen REQUIRED)
#find_package(CUDA REQUIRED)
#include_directories(${CUDA_INCLUDE_DIRS})

macro(add_code name)
   add_executable(${name} ${name}.cpp)
   target_link_libraries(${name} "${TORCH_LIBRARIES}")

   #target_link_libraries(${name} "${CUDA_LIBRARIES}")
   #target_link_libraries(${name} "${ATEN_LIBRARIES}")
   #target_link_libraries(${name} dl)
   #set_property(TARGET ${name} PROPERTY CXX_STANDARD 11)
endmacro()

add_code(test)
add_code(mnist)
