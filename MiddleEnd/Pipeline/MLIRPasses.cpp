// MiddleEnd/Pipeline/MLIRPasses.cpp
#include "MLIRPasses.hpp"

#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

namespace tcc {
namespace mlir_gen {

MLIRPasses::MLIRPasses() = default;
MLIRPasses::~MLIRPasses() = default;

void MLIRPasses::addLoweringPasses(mlir::PassManager& pm) {
    // 1. Linalg -> SCF (циклы)
    pm.addPass(mlir::createConvertLinalgToLoopsPass());

    // 2. SCF -> CFG (базовые блоки)
    pm.addPass(mlir::createConvertSCFToCFPass());

    // 3. Убираем временные касты
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

bool MLIRPasses::runLoweringPipeline(mlir::ModuleOp module) {
    mlir::PassManager pm(module.getContext());
    pm.enableVerifier(true);

    addLoweringPasses(pm);

    if (mlir::failed(pm.run(module))) {
        llvm::errs() << "Error: Lowering pipeline failed\n";
        return false;
    }

    return true;
}

} // namespace mlir_gen
} // namespace tcc
