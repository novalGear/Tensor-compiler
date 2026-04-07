set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# 1. Code Generation (Ruby Scripts)
# ==============================================================================

# Пути к скриптам и конфигурации
set(GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/generate_nodes.rb")
set(GEN_LIB "${CMAKE_CURRENT_SOURCE_DIR}/generator_lib.rb")
set(GEN_JSON "${CMAKE_CURRENT_SOURCE_DIR}/ops_def.json")

# Пути к генерируемым файлам
set(GEN_OUTPUT_HPP "${CMAKE_CURRENT_SOURCE_DIR}/Graph/graph_gen.hpp")
set(GEN_OUTPUT_INL "${CMAKE_CURRENT_SOURCE_DIR}/Graph/graph_gen_parser.inl")
set(GEN_OUTPUT_UTILS "${CMAKE_CURRENT_SOURCE_DIR}/Graph/graph_gen_utils.inl")

# Список всех выходных файлов
set(GENERATED_FILES
    ${GEN_OUTPUT_HPP}
    ${GEN_OUTPUT_INL}
    ${GEN_OUTPUT_UTILS}
)

# Правило генерации
add_custom_command(
    OUTPUT ${GENERATED_FILES}
    COMMAND ruby ${GEN_SCRIPT}
    DEPENDS ${GEN_SCRIPT} ${GEN_LIB} ${GEN_JSON}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running Ruby generator (ops_def.json -> C++ code)..."
    VERBATIM
)

add_custom_target(generate_node_code ALL
    DEPENDS ${GENERATED_FILES}
)

set_source_files_properties(${GENERATED_FILES} PROPERTIES GENERATED TRUE)

# ==============================================================================
# 2. Зависимости
# ==============================================================================
find_package(Protobuf REQUIRED)

# ==============================================================================
# 3. Работа с ONNX (FetchContent)
# ==============================================================================

include(FetchContent)
FetchContent_Declare(
  onnx
  GIT_REPOSITORY https://github.com/onnx/onnx.git
  GIT_TAG        v1.20.1
)

set(ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ONNX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_ONNX_PYTHON OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(onnx)

message(STATUS "onnx_SOURCE_DIR: ${onnx_SOURCE_DIR}")
message(STATUS "onnx_BINARY_DIR: ${onnx_BINARY_DIR}")

set(ONNX_GEN_PROTO_DIR ${onnx_BINARY_DIR})
message(STATUS "ONNX_GEN_PROTO_DIR: ${ONNX_GEN_PROTO_DIR}")

# Файлы, которые генерируются
set(ONNX_GENERATED_SRCS
    ${ONNX_GEN_PROTO_DIR}/onnx/onnx-ml.pb.cc
    ${ONNX_GEN_PROTO_DIR}/onnx/onnx-operators-ml.pb.cc
    ${ONNX_GEN_PROTO_DIR}/onnx/onnx-data.pb.cc
)

add_library(onnx_proto_lib STATIC ${ONNX_GENERATED_SRCS})

target_include_directories(onnx_proto_lib PUBLIC
    ${ONNX_GEN_PROTO_DIR}
    ${onnx_SOURCE_DIR}
    ${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(onnx_proto_lib PUBLIC
    ${Protobuf_LIBRARIES}
)

# Добавляем зависимость от генерации proto
add_dependencies(onnx_proto_lib onnx_proto)

# ==============================================================================
# 4. Библиотека Frontend (вместо исполняемого файла)
# ==============================================================================

set(FRONTEND_SOURCES
    Graph/graph.cpp
    Graph/graph_utils.cpp
    Graph/convertion.cpp
    Graph/topological_sort.cpp
)

add_library(tcc_frontend STATIC ${FRONTEND_SOURCES})

add_dependencies(tcc_frontend generate_node_code)

target_include_directories(tcc_frontend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Graph
    ${ONNX_GEN_PROTO_DIR}
    ${onnx_SOURCE_DIR}
    ${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(tcc_frontend PRIVATE
    onnx_proto_lib
    ${Protobuf_LIBRARIES}
)

# ==============================================================================
# 5. Опционально: тестовый исполняемый файл (если нужен)
# ==============================================================================

# Если нужен отдельный тестовый бинарник для фронтенда:
# add_executable(frontend_test main.cpp)
# target_link_libraries(frontend_test PRIVATE tcc_frontend)

message(STATUS "Frontend configured. Protos generated in: ${ONNX_GEN_PROTO_DIR}")
