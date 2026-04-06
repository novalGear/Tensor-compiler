# MiddleEnd/Pipeline/CMakeLists.txt

add_library(tcc_pipeline STATIC
    MLIRPasses.cpp
)

target_include_directories(tcc_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(tcc_pipeline
    PRIVATE
    MLIRIR
    MLIRPass
    MLIRFuncToLLVM
    MLIRArithToLLVM
    MLIRSCFToControlFlow
    MLIRTensorToLinalg
    MLIRReconcileUnrealizedCasts
    LLVMSupport
)
