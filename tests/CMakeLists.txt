# tests/CMakeLists.txt

# Включаем тестирование
# enable_testing()

# ================================================================
# Тесты Graph
# ================================================================

add_executable(tcc_graph_tests
    test_graph_methods.cpp
)

target_link_libraries(tcc_graph_tests
    PRIVATE
    tcc_frontend
    GTest::gtest
    GTest::gtest_main
    ${MLIR_LIBS}
    ${LLVM_LIBS}
)

# ================================================================
# Тесты Type Converter
# ================================================================
add_executable(tcc_type_converter_tests
    test_type_converter.cpp
)

target_link_libraries(tcc_type_converter_tests
    PRIVATE
    tcc_frontend
    tcc_middleend
    GTest::gtest
    GTest::gtest_main
    ${MLIR_LIBS}
    ${LLVM_LIBS}
)

# ================================================================
# Тесты MLIR Generator
# ================================================================
add_executable(tcc_mlir_generator_tests
    test_mlir_generator.cpp
)

target_link_libraries(tcc_mlir_generator_tests
    PRIVATE
    tcc_frontend
    tcc_middleend
    GTest::gtest
    GTest::gtest_main
    ${MLIR_LIBS}
    ${LLVM_LIBS}
)

# ================================================================
# Тесты топологической сортировки
# ================================================================
add_executable(tcc_topological_tests
    test_topological_sort.cpp
)

target_link_libraries(tcc_topological_tests
    PRIVATE
    tcc_frontend
    tcc_middleend
    GTest::gtest
    GTest::gtest_main
    ${MLIR_LIBS}
    ${LLVM_LIBS}
)

#=================================================================
# Тесты эмиттеров
#=================================================================

add_executable(tcc_emitter_tests
    test_const_emitter.cpp
    test_add_emitter.cpp
    test_mul_emitter.cpp
    test_ReLU_emitter.cpp
    test_matmul_emitter.cpp
)

target_link_libraries(tcc_emitter_tests
    PRIVATE
    tcc_frontend
    tcc_middleend
    GTest::gtest
    GTest::gtest_main
    ${MLIR_LIBS}
    ${LLVM_LIBS}
)

add_executable(tcc_integration_tests
    test_integration.cpp
)

target_link_libraries(tcc_integration_tests
    PRIVATE
    tcc_frontend
    tcc_middleend
    GTest::gtest
    GTest::gtest_main
    ${MLIR_LIBS}
    ${LLVM_LIBS}
)


# ================================================================
# Регистрация всех тестов в CTest
# ================================================================
include(GoogleTest)

gtest_discover_tests(tcc_graph_tests)
gtest_discover_tests(tcc_type_converter_tests)
gtest_discover_tests(tcc_mlir_generator_tests)
gtest_discover_tests(tcc_topological_tests)
gtest_discover_tests(tcc_emitter_tests)
gtest_discover_tests(tcc_integration_tests)
