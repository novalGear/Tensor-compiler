// MiddleEnd/Pipeline/MLIRPasses.hpp
#pragma once

#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinOps.h"

namespace tcc {
namespace mlir_gen {

class MLIRPasses {
public:
    MLIRPasses();
    ~MLIRPasses();

    // Базовое понижение MLIR до LLVM диалекта
    bool runLoweringPipeline(mlir::ModuleOp module);

private:
    void addLoweringPasses(mlir::PassManager& pm);
};

} // namespace mlir_gen
} // namespace tcc
