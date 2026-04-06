# tests/CMakeLists.txt

# ================================================================
# Тесты Graph (не требуют MLIR)
# ================================================================
add_executable(tcc_graph_tests
    test_graph_methods.cpp
)

target_link_libraries(tcc_graph_tests
    PRIVATE
    tcc_frontend
    GTest::gtest
    GTest::gtest_main
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
    MLIRIR
    MLIRParser
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTensorDialect
    LLVMSupport
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
    MLIRIR
    MLIRParser
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTensorDialect
    LLVMSupport
)

# ================================================================
# Тесты топологической сортировки (не требуют MLIR)
# ================================================================
add_executable(tcc_topological_tests
    test_topological_sort.cpp
)

target_link_libraries(tcc_topological_tests
    PRIVATE
    tcc_frontend
    GTest::gtest
    GTest::gtest_main
)

# ================================================================
# Тесты эмиттеров
# ================================================================
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
    MLIRIR
    MLIRParser
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTensorDialect
    LLVMSupport
)

# ================================================================
# Интеграционные тесты
# ================================================================
add_executable(tcc_integration_tests
    test_integration.cpp
)

target_link_libraries(tcc_integration_tests
    PRIVATE
    tcc_frontend
    tcc_middleend
    GTest::gtest
    GTest::gtest_main
    MLIRIR
    MLIRParser
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTensorDialect
    LLVMSupport
)

# ================================================================
# Тесты Passes
# ================================================================
add_executable(tcc_mlir_passes_tests
    test_mlir_passes.cpp
)

target_link_libraries(tcc_mlir_passes_tests
    PRIVATE
    tcc_frontend
    tcc_middleend
    GTest::gtest
    GTest::gtest_main
    MLIRIR
    MLIRParser
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTensorDialect
    MLIRPass
    LLVMSupport
)

# ================================================================
# Регистрация всех тестов в CTest
# ================================================================
enable_testing()
include(GoogleTest)

gtest_discover_tests(tcc_graph_tests)
gtest_discover_tests(tcc_type_converter_tests)
gtest_discover_tests(tcc_mlir_generator_tests)
gtest_discover_tests(tcc_topological_tests)
gtest_discover_tests(tcc_emitter_tests)
gtest_discover_tests(tcc_integration_tests)
gtest_discover_tests(tcc_mlir_passes_tests)
