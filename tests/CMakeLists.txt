# tests/CMakeLists.txt (создать, если нет)
add_executable(tcc_tests
    test_type_converter.cpp
    test_topological_sort.cpp
    test_mlir_generator.cpp
)

target_link_libraries(tcc_tests
    PRIVATE
    tcc_frontend
    tcc_middleend
    GTest::gtest
    GTest::gtest_main
    ${MLIR_LIBS}
    ${LLVM_LIBS}
)

include(GoogleTest)
gtest_discover_tests(tcc_tests)
