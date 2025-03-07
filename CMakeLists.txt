cmake_minimum_required(VERSION 3.5)



set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")

include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES "86")
# set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v --generate-line-info")

# find_package(Python REQUIRED)

# set(SPARSE_DATA_FILE "${CMAKE_CURRENT_SOURCE_DIR}/sparse6")
# set(GENERATED_CUDA_FILE "${CMAKE_CURRENT_SOURCE_DIR}/officers_loop.inc")

# add_custom_command(
#     OUTPUT ${GENERATED_CUDA_FILE}
#     COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_loop.py
#             ${SPARSE_DATA_FILE} ${GENERATED_CUDA_FILE}
#     DEPENDS ${SPARSE_DATA_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_loop.py
#     COMMENT "Generating CUDA loop from sparse values"
# )

add_custom_target(generate_loop DEPENDS ${GENERATED_CUDA_FILE})

add_executable(officers_kernel officers_kernel.cu)
add_dependencies(officers_kernel generate_loop)

add_executable(officers_naive officers_naive.cpp)

project(officers)
