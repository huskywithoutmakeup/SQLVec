cmake_minimum_required(VERSION 3.10)
project(ANNSonSQL)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER g++)

# Include DuckDB header file path
include_directories(${PROJECT_SOURCE_DIR}/libduckdb)
# Link DuckDB library file path
link_directories(${PROJECT_SOURCE_DIR}/libduckdb)

# # Set LibTorch path
# set(Torch_DIR "${PROJECT_SOURCE_DIR}/libtorch/share/cmake/Torch")
# # Find Torch package (this defines variables like Torch_LIBRARIES)
# find_package(Torch REQUIRED)

# Set C++ compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=native -mavx512vnni")

# Add executable files
add_executable(SQLVec SQLVec.cpp)
add_executable(SQLVec_opti_nopq SQLVec_opti_nopq.cpp)
add_executable(SQLVec_opti_pq SQLVec_opti_pq.cpp)

# target_link_libraries(udf ${TORCH_LIBRARIES})

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
# Link DuckDB library
target_link_libraries(SQLVec ${PROJECT_SOURCE_DIR}/libduckdb/libduckdb.so)
target_link_libraries(SQLVec_opti_nopq ${PROJECT_SOURCE_DIR}/libduckdb/libduckdb.so)
target_link_libraries(SQLVec_opti_pq ${PROJECT_SOURCE_DIR}/libduckdb/libduckdb.so)

# DuckDB can be compiled with C++, but it must be compatible with the CXX11 ABI