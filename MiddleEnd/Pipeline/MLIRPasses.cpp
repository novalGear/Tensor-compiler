// MiddleEnd/Pipeline/MLIRPasses.cpp
#include "MLIRPasses.hpp"

#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"


#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
// #include "mlir/Conversion/LinalgToLoops/LinalgToLoopsPass"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "mlir/IR/MLIRContext.h"

#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"


namespace tcc {
namespace mlir_gen {

MLIRPasses::MLIRPasses() = default;
MLIRPasses::~MLIRPasses() = default;

void MLIRPasses::addLoweringPasses(mlir::PassManager& pm) {

    // pm.addPass(mlir::createConvertTensorToLinalgPass());    // tensor -> linalg
    // pm.addPass(mlir::createConvertLinalgToLoopsPass());     // linalg -> loops
    // // pm.addPass(mlir::createCanonicalizerPass());            // упрощение IR

    // // pm.addPass(mlir::createConvertSCFToCFPass());           // scf -> cf
    // // // pm.addPass(mlir::createConvertFuncToLLVMPass());        // func -> llvm.func
    // // pm.addPass(mlir::createArithToLLVMConversionPass());    // arith -> llvm ops
    // // pm.addPass(mlir::createReconcileUnrealizedCastsPass()); // склеивание cast'ов
    // // pm.addPass(mlir::createCanonicalizerPass());            // финальная чистка

    // 1️⃣ Tensor → Linalg
    pm.addPass(mlir::createConvertTensorToLinalgPass());
    pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass()); // empty tensors to alloc
    pm.addPass(mlir::createCanonicalizerPass());

    // 2️⃣ Tensor → MemRef (Bufferization)
    mlir::bufferization::OneShotBufferizationOptions bufOpts;
    bufOpts.bufferizeFunctionBoundaries = true;
    bufOpts.setFunctionBoundaryTypeConversion(mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
    bufOpts.allowReturnAllocs = true;  
    bufOpts.allowUnknownOps = true;    
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));

    // pm.addPass(mlir::bufferization::createBufferDeallocationPass());
    pm.addPass(mlir::createCanonicalizerPass());

    // 3️⃣ Linalg → SCF циклы
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createCanonicalizerPass());

    // 4️⃣ SCF → Control Flow (cf.br, cf.cond_br)
    pm.addPass(mlir::createConvertSCFToCFPass());

    // 5️⃣ Финальный перевод в LLVM Dialect
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

    // 6️⃣ Очистка "мостовых" cast'ов и dead code
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

}

bool MLIRPasses::runLoweringPipeline(mlir::ModuleOp module) {
    // Registering LLVM dialect
    mlir::MLIRContext *context = module->getContext();
    context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context->loadDialect<mlir::memref::MemRefDialect>();
    context->loadDialect<mlir::bufferization::BufferizationDialect>();

    mlir::PassManager pm(module.getContext());
    pm.enableVerifier(true);

    // adding passes
    addLoweringPasses(pm);

    // running passes

    pm.getContext()->disableMultithreading();
    pm.enableIRPrinting();

    if (mlir::failed(pm.run(module))) {
        llvm::errs() << "Error: Lowering pipeline failed\n";
        return false;
    }

    module->print(llvm::outs());

    // Exporting to llvm::Module
    llvm::LLVMContext llvmCtx;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmCtx, "lowered_module");
    if (!llvmModule) {
        llvm::errs() << " Translation to LLVM IR failed\n";
        return false;
    }

    // target triple & data layout (for x86-64 linux)
    llvmModule->setTargetTriple("x86_64-unknown-linux-gnu");
    llvmModule->setDataLayout(
        "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128");

    llvm::outs() << "\n Successfully converted to LLVM IR:\n";
    llvmModule->print(llvm::outs(), nullptr);

    return true;
}

} // namespace mlir_gen
} // namespace tcc
