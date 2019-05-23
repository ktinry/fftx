# FFTX Copyright (c) 2019, The Regents of the University of California, through 
# Lawrence Berkeley National Laboratory (subject to receipt of any required 
# approvals from the U.S. Dept. of Energy), Carnegie Mellon University and 
# SpiralGen, Inc.  All rights reserved.
# 
# If you have questions about your rights to use or distribute this software, 
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
# 
# NOTICE.  This Software was developed under funding from the U.S. Department of
# Energy and the U.S. Government  consequently  retains certain rights. As such, 
# the U.S. Government has been granted for itself and others acting on its 
# behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software 
# to reproduce, distribute copies to the public, prepare derivative works, and 
# perform publicly and display publicly, and to permit others to do so.

# specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required(VERSION 3.8...3.14)
if (${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(FFTX 
        VERSION 1.0.0.0
		DESCRIPTION "ECP FFTX Project"
		LANGUAGES C CXX)


# prevent building directly into the source tree
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" FFTX_COMPILE_INPLACE)
if (FFTX_COMPILE_INPLACE)
	message( FATAL_ERROR "Compiling ${PROJECT_NAME} with CMake requires an out-of-source build. To proceed:
    rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_SOURCE_DIR}
    mkdir build
    cd build
    cmake ..
    make" )
endif()

set(FFTX_SRC ${FFTX_SOURCE_DIR}/src)
set(FFTX_INC ${FFTX_SOURCE_DIR}/include/fftx)
set(FFTX_CONFIG_DIR ${FFTX_SOURCE_DIR}/config)
set(FFTX_GEN_INC ${FFTX_BINARY_DIR}/include/fftx)

set(CMAKE_C_STANDARD 99)

include_directories(${FFTX_INC} ${FFTX_GEN_INC})

# locate and connect FFTW
include("${CMAKE_CURRENT_SOURCE_DIR}/config/CMakeIncludes/locate_fftw3.cmake")

set(FFTW_INCLUDE_DIR ${LOCAL_FFTW_INCLUDE_DIR}  CACHE PATH   "directory containing fftw3.h")
set(FFTW_LIBRARY     ${LOCAL_FFTW_LIBRARY}      CACHE PATH   "FFTW library")


message(STATUS "FFTW_INCLUDE_DIR = ${FFTW_INCLUDE_DIR}")
message(STATUS "FFTW_LIBRARY = ${FFTW_LIBRARY}")

include_directories(${FFTW_INCLUDE_DIR})
link_libraries(${FFTW_LIBRARY})

# Check for the git commit hash, if using a git repo
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    find_package(Git)
    if (Git_FOUND)
        execute_process(COMMAND ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
                        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                        OUTPUT_VARIABLE   FFTX_GIT_HASH)
    endif()
endif()
if (NOT FFTX_GIT_HASH)
   set(FFTX_GIT_HASH "Not found")
endif()

if ((NOT DEFINED CMAKE_BUILD_TYPE) OR (NOT CMAKE_BUILD_TYPE))
    set(CMAKE_BUILD_TYPE Debug)
endif()

configure_file (
    ${FFTX_CONFIG_DIR}/fftx_build_info.h.in
    ${FFTX_GEN_INC}/fftx_build_info.h
)

add_subdirectory(test)
add_subdirectory(examples)

enable_testing()

