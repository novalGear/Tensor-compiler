# MiddleEnd/CMakeLists.txt

add_subdirectory(MLIR)
add_subdirectory(Pipeline)

add_library(tcc_middleend INTERFACE)

# Добавляем пути к Frontend заголовочным файлам
target_include_directories(tcc_middleend INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/..          # корень проекта
    ${CMAKE_CURRENT_SOURCE_DIR}/../Frontend # папка Frontend
    ${CMAKE_CURRENT_SOURCE_DIR}/../Frontend/Graph  # папка Graph
    ${CMAKE_CURRENT_BINARY_DIR}/../../_deps/onnx-build
    ${CMAKE_CURRENT_BINARY_DIR}/../../_deps/onnx-build/onnx
    ${CMAKE_CURRENT_BINARY_DIR}/../../_deps/onnx-src
)

target_link_libraries(tcc_middleend INTERFACE tcc_mlir)
