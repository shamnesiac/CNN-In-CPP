cmake_minimum_required(VERSION 3.10)
project(CNNCPP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Enable OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Set OpenMP environment variables during build
set(ENV{OMP_SCHEDULE} "dynamic")
set(ENV{OMP_PROC_BIND} "true")
set(ENV{OMP_PLACES} "cores")
set(ENV{OMP_MAX_ACTIVE_LEVELS} "3")

# Enable all optimizations
if(MSVC)
    # MSVC-specific optimizations
    set(OPTIMIZATION_FLAGS "/O2 /Oi /GL /Gy /Ot /arch:AVX2 /fp:fast /Qpar /Qpar-report:2")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
else()
    # GCC/Clang optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(OPTIMIZATION_FLAGS "-march=native -mtune=native -mavx2 -mfma -ffast-math -funroll-loops -flto -fomit-frame-pointer -finline-functions -ftree-vectorize")
endif()

# Set the optimization flags for Release build
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPTIMIZATION_FLAGS}")

# Add source files
set(COMMON_SOURCES
    src/cnn.cpp
    src/evaluate.cpp
    src/mlp.cpp
)

# Add header files
set(HEADERS
    include/cnn.hpp
    include/evaluate.hpp
    include/mlp.hpp
)

# Function to set common properties for executables
function(set_executable_properties target_name)
    # Add include directories
    target_include_directories(${target_name} PRIVATE include)
    
    # Link OpenMP
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
    # Set build type specific properties
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

# Create executables
add_executable(${PROJECT_NAME} src/cnn_main.cpp ${COMMON_SOURCES} ${HEADERS})
add_executable(evaluate src/evaluate_main.cpp ${COMMON_SOURCES} ${HEADERS})
add_executable(mlp_train src/mlp_main.cpp ${COMMON_SOURCES} ${HEADERS})
add_executable(evaluate_mlp src/evaluate_mlp_main.cpp ${COMMON_SOURCES} ${HEADERS})

# Set properties for all executables
set_executable_properties(${PROJECT_NAME})
set_executable_properties(evaluate)
set_executable_properties(mlp_train)
set_executable_properties(evaluate_mlp)

# Create data directory in build
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/data
)

# Add custom targets for running the programs
add_custom_target(run_train
    COMMAND ${CMAKE_COMMAND} -E env 
        OMP_NUM_THREADS=8
        OMP_SCHEDULE=dynamic 
        OMP_PROC_BIND=true 
        OMP_PLACES=cores 
        $<TARGET_FILE:${PROJECT_NAME}> --train-size 60000 --test-size 10000 --epochs 5 --batch-size 64
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running CNN training with optimal settings"
)

add_custom_target(run_evaluate
    COMMAND ${CMAKE_COMMAND} -E env 
        OMP_NUM_THREADS=8
        OMP_SCHEDULE=dynamic 
        OMP_PROC_BIND=true 
        OMP_PLACES=cores 
        $<TARGET_FILE:evaluate> 10000
    DEPENDS evaluate
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running CNN evaluation with optimal settings"
)

add_custom_target(run_mlp
    COMMAND ${CMAKE_COMMAND} -E env 
        OMP_NUM_THREADS=8
        OMP_SCHEDULE=dynamic 
        OMP_PROC_BIND=true 
        OMP_PLACES=cores 
        $<TARGET_FILE:mlp_train> --train-size 60000 --test-size 10000 --epochs 5 --batch-size 64
    DEPENDS mlp_train
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running MLP training with optimal settings"
)

add_custom_target(run_evaluate_mlp
    COMMAND ${CMAKE_COMMAND} -E env 
        OMP_NUM_THREADS=8
        OMP_SCHEDULE=dynamic 
        OMP_PROC_BIND=true 
        OMP_PLACES=cores 
        $<TARGET_FILE:evaluate_mlp> 10000
    DEPENDS evaluate_mlp
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running MLP evaluation with optimal settings"
)

# Print configuration summary
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "OpenMP Found: ${OpenMP_CXX_FOUND}")
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS_RELEASE}")

# Print usage instructions
message(STATUS "")
message(STATUS "Build instructions:")
message(STATUS "  1. mkdir build")
message(STATUS "  2. cd build")
message(STATUS "  3. cmake ..")
message(STATUS "  4. make -j$(nproc)")
message(STATUS "")
message(STATUS "Run instructions:")
message(STATUS "  - Training:   make run_train")
message(STATUS "  - Evaluation: make run_evaluate")
message(STATUS "  - MLP Training: make run_mlp")
message(STATUS "  - MLP Evaluation: make run_evaluate_mlp")
message(STATUS "") 