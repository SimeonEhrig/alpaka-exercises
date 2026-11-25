#
# Copyright 2025 Tapish Narwal
# SPDX-License-Identifier: ISC
#

cmake_minimum_required(VERSION 3.25)

project(device_selection LANGUAGES CXX VERSION 1.0)

# force enable CUDA backend
set(alpaka_DEP_CUDA ON)

include(FetchContent)
FetchContent_Declare(alpaka GIT_REPOSITORY https://github.com/alpaka-group/alpaka.git GIT_TAG 3.0.0)
FetchContent_MakeAvailable(alpaka)

# name of the executable
set(_TARGET_NAME alpaka_task)
# Create the executable target from the source file
add_executable(${_TARGET_NAME} main.cpp)

# Link the alpaka library to the executable
target_link_libraries(${_TARGET_NAME} PRIVATE alpaka::alpaka)

# Disable deprecation warning on Nvidia Tesla P100
target_compile_options(${_TARGET_NAME} PRIVATE  "-Wno-deprecated-gpu-targets")

# Finalize the alpaka configuration for this target
# This sets up backend-specific compiler flags and dependencies
alpaka_finalize(${_TARGET_NAME})
