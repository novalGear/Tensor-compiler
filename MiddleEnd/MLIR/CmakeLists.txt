# MiddleEnd/MLIR/CMakeLists.txt

set(MLIR_SOURCES
    TypeConverter.cpp
    MLIRGenerator.cpp
    OperationEmitters/AddEmitter.cpp
    OperationEmitters/MulEmitter.cpp
    OperationEmitters/ConstantEmitter.cpp
    OperationEmitters/ReLUEmitter.cpp
    OperationEmitters/MatMulEmitter.cpp
)

add_library(tcc_mlir STATIC ${MLIR_SOURCES})

target_include_directories(tcc_mlir PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Frontend
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Frontend/Graph
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(tcc_mlir
    PRIVATE
    tcc_frontend
    tcc_pipeline
    MLIRIR
    MLIRParser
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTensorDialect
    MLIRPass
    LLVMSupport
)
