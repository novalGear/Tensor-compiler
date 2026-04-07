# Корневой CMakeLists.txt для WSL
cmake_minimum_required(VERSION 3.26)

project(TensorCompiler
  LANGUAGES C CXX
  VERSION 0.1
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ================================================================
# Plog debugging library
# ================================================================

add_subdirectory(3rd-party/plog)

# ================================================================
# Автоматический поиск LLVM/MLIR в WSL типовых путях
# ================================================================

set(LLVM_VERSIONS 18 17 16 15 14)

foreach(VER ${LLVM_VERSIONS})
    set(LLVM_PATH "/usr/lib/llvm-${VER}")
    if(EXISTS ${LLVM_PATH})
        list(APPEND CMAKE_PREFIX_PATH ${LLVM_PATH})
        list(APPEND CMAKE_PREFIX_PATH "${LLVM_PATH}/lib/cmake")
        message(STATUS "Found LLVM ${VER} at ${LLVM_PATH}")
    endif()
endforeach()

list(APPEND CMAKE_PREFIX_PATH
    "/usr/lib/llvm"
    "/usr/local/lib/llvm"
    "/usr/lib/llvm-17/lib/cmake"
)

# ================================================================
# Поиск пакетов
# ================================================================
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(Protobuf REQUIRED)

message(STATUS "LLVM_DIR: ${LLVM_DIR}")
message(STATUS "MLIR_DIR: ${MLIR_DIR}")
message(STATUS "Protobuf_DIR: ${Protobuf_DIR}")

# Добавляем пути к include и lib
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
link_directories(${MLIR_LIBRARY_DIRS})

add_definitions(${LLVM_DEFINITIONS})

# ================================================================
# Поддиректории
# ================================================================
add_subdirectory(Frontend)
add_subdirectory(MiddleEnd)
add_subdirectory(googletest)

enable_testing()
add_subdirectory(tests)

# ================================================================
# Глобальные пути для include
# ================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Frontend
    ${CMAKE_CURRENT_SOURCE_DIR}/Frontend/Graph
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/onnx-build
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/onnx-build/onnx
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/onnx-src
)

# ================================================================
# Драйвер
# ================================================================
add_executable(tcc_driver driver.cpp)

target_link_libraries(tcc_driver
    PRIVATE
    tcc_frontend
    tcc_middleend
    ${MLIR_LIBS}
    ${LLVM_LIBS}
    ${PROTOBUF_LIBRARIES}
    plog::plog
)
