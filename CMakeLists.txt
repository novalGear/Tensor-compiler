cmake_minimum_required(VERSION 3.26)
project(TensorCompiler
  LANGUAGES CXX
  VERSION 0.1
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(Frontend)
add_subdirectory(googletest)

enable_testing()

# include(GoogleTest)
# gtest_discover_tests(${unit_test_exec_name}
#     PROPERTIES
#         LABELS "parser_end2end"
# )
