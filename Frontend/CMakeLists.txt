set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Google Protobuf
find_package(Protobuf REQUIRED)

# Onnx
include(FetchContent)
FetchContent_Declare(
  onnx
  GIT_REPOSITORY https://github.com/onnx/onnx.git
  GIT_TAG        v1.20.1 # stable version on 25.02.2026
)
FetchContent_MakeAvailable(onnx)


set(ONNX_PROTO_FILES
    ${onnx_SOURCE_DIR}/onnx/onnx.proto
    ${onnx_SOURCE_DIR}/onnx/onnx-ml.proto
    ${onnx_SOURCE_DIR}/onnx/onnx-data.proto
    ${onnx_SOURCE_DIR}/onnx/onnx-operators.proto
)

# 4. ГЕНЕРАЦИЯ КОДА (НОВЫЙ СПОСОБ)
# Создаем кастомный таргет для генерации, чтобы управлять зависимостями
protobuf_generate(
    LANGUAGE cpp
    OUT_VAR ONNX_PROTO_SRCS
    PROTOS ${ONNX_PROTO_FILES}
    PROTOC_OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/onnx_proto
    IMPORT_DIRS ${onnx_SOURCE_DIR} # Критично: где искать импорты внутри proto файлов
    # DEPENDS protoc # Убеждаемся, что protoc собран/найден
)


# onnx proto lib
add_library(onnx_proto_lib ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})
target_include_directories(onnx_proto_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/onnx_proto
    ${onnx_SOURCE_DIR}
    ${Protobuf_INCLUDE_DIRS}
)

add_executable(frontend main.cpp)
target_link_libraries(onnx_proto_lib PUBLIC ${Protobuf_LIBRARIES})


target_link_libraries(frontend PRIVATE onnx_proto_lib)
