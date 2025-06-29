# CMake minimum version
cmake_minimum_required(VERSION 3.15)

# Project name
project(SNN_Project CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the OpenMP package (required for parallelization)
find_package(OpenMP REQUIRED)

# Add the executable and list all source files
add_executable(SNN_Project
    main.cpp
    config.cpp
    datalogger.cpp
    experiment.cpp
    learning_rules.cpp
    network.cpp
    neuron.cpp
    synapse.cpp
)

# Link the OpenMP libraries to our executable
# This is the modern, portable way to enable OpenMP
target_link_libraries(SNN_Project PRIVATE OpenMP::OpenMP_CXX)

# Tell the compiler where to find headers (the current directory)
target_include_directories(SNN_Project PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Optional: Add compiler flags for optimization in release builds
# target_compile_options(SNN_Project PRIVATE $<$<CONFIG:Release>:-O3>)