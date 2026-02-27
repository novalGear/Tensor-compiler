set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# 1. Зависимости
# ==============================================================================
find_package(Protobuf REQUIRED)

# ==============================================================================
# 2. Скачивание и Генерация ONNX
# ==============================================================================
include(FetchContent)
FetchContent_Declare(
  onnx
  GIT_REPOSITORY https://github.com/onnx/onnx.git
  GIT_TAG        v1.20.1
)

# Отключаем лишнее, но разрешаем сборку их proto либы, если нужно,
# хотя мы будем делать свою.
set(ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ONNX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_ONNX_PYTHON OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(onnx)

# Путь, куда будут сгенерированы файлы
set(ONNX_PROTO_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnx_proto")
file(MAKE_DIRECTORY ${ONNX_PROTO_OUTPUT_DIR})

set(ONNX_PROTO_FILES
    ${onnx_SOURCE_DIR}/onnx/onnx.proto
    ${onnx_SOURCE_DIR}/onnx/onnx-ml.proto
    ${onnx_SOURCE_DIR}/onnx/onnx-data.proto
    ${onnx_SOURCE_DIR}/onnx/onnx-operators.proto
)

# Генерируем файлы
protobuf_generate(
    LANGUAGE cpp
    OUT_VAR ONNX_PROTO_SRCS_GENERATED # Переименовали, чтобы не путаться
    PROTOS ${ONNX_PROTO_FILES}
    PROTOC_OUT_DIR ${ONNX_PROTO_OUTPUT_DIR}
    IMPORT_DIRS ${onnx_SOURCE_DIR}
)

# === ИСПРАВЛЕНИЕ ПУТЕЙ ===
set(ONNX_PROTO_SRCS_FIXED)
set(ONNX_PROTO_HDRS_FIXED)

foreach(proto_file ${ONNX_PROTO_FILES})
    get_filename_component(fname ${proto_file} NAME_WE)
    list(APPEND ONNX_PROTO_SRCS_FIXED "${ONNX_PROTO_OUTPUT_DIR}/onnx/${fname}.pb.cc")
    list(APPEND ONNX_PROTO_HDRS_FIXED "${ONNX_PROTO_OUTPUT_DIR}/onnx/${fname}.pb.h")
endforeach()

# ==============================================================================
# 3. Библиотека ONNX Proto
# ==============================================================================

# Создаем библиотеку. CMake автоматически поймет, что эти .cc файлы требуют генерации,
# так как они являются выходом правила protobuf_generate, которое мы вызвали выше.
add_library(onnx_proto_lib STATIC ${ONNX_PROTO_SRCS_FIXED} ${ONNX_PROTO_HDRS_FIXED})

# !!! УДАЛИЛИ СТРОКУ add_dependencies, так как она вызывала ошибку !!!
# add_dependencies(onnx_proto_lib ${ONNX_PROTO_SRCS_GENERATED}) <-- ЭТО БЫЛО ОШИБКОЙ

target_include_directories(onnx_proto_lib PUBLIC
    ${ONNX_PROTO_OUTPUT_DIR}
    ${onnx_SOURCE_DIR}
    ${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(onnx_proto_lib PUBLIC
    ${Protobuf_LIBRARIES}
)

# ==============================================================================
# 4. Основной исполняемый файл
# ==============================================================================

set(FRONTEND_SOURCES
    main.cpp
    # Graph/graph.cpp
    Graph/graph_utils.cpp
    Graph/convertion.cpp
    Graph/load_from_proto.cpp
)

add_executable(frontend ${FRONTEND_SOURCES})

target_include_directories(frontend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Graph
)

target_link_libraries(frontend PRIVATE
    onnx_proto_lib
)

message(STATUS "Frontend configured. Protos generated in: ${ONNX_PROTO_OUTPUT_DIR}")
