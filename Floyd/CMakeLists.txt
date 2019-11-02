cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(CUDA2 VERSION 1.0 LANGUAGES C CUDA)
add_definitions(-DCUDA_ENABLED)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

add_executable(Floyd main.cu)
target_include_directories(Floyd PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(Floyd PRIVATE ${CUDA_INCLUDE_DIRECTORIES})
target_link_libraries(Floyd ${MPI_C_LIBRARIES})
target_link_libraries(Floyd ${CUDA_LIBRARIES})
