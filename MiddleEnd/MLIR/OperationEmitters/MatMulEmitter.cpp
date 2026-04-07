// MatMulEmitter.cpp
#include "MatMulEmitter.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "plog/Log.h"

#include <iostream>

namespace tcc {
namespace mlir_gen {

mlir::Value MatMulEmitter::emit(const std::vector<mlir::Value>& inputs,
                                 const std::vector<std::string>& outputNames,
                                 const std::vector<size_t>& outputDims) {

    PLOG_DEBUG << "[MatMulEmitter] START emit\n";
    PLOG_DEBUG << "[MatMulEmitter] outputNames[0] = " << outputNames[0] << "\n";
    PLOG_DEBUG << "[MatMulEmitter] outputDims size = " << outputDims.size() << "\n";

    auto loc = mlir::UnknownLoc::get(builder.getContext());

    if (inputs.size() != 2) {
        llvm::errs() << "MatMul operation requires exactly 2 inputs\n";
        return nullptr;
    }

    if (outputNames.empty()) {
        llvm::errs() << "MatMul requires at least one output name\n";
        return nullptr;
    }

    auto lhs = inputs[0];
    auto rhs = inputs[1];

    auto lhsType = lhs.getType().cast<mlir::RankedTensorType>();
    auto rhsType = rhs.getType().cast<mlir::RankedTensorType>();
    auto elementType = lhsType.getElementType();

    // Создание типа результата
    llvm::SmallVector<int64_t> mlirDims;
    for (auto dim : outputDims) {
        mlirDims.push_back(static_cast<int64_t>(dim));
    }
    auto resultType = mlir::RankedTensorType::get(mlirDims, elementType);

    // Пустой тензор для результата
    auto empty = builder.create<mlir::tensor::EmptyOp>(loc, mlirDims, elementType);

    int rank = resultType.getRank();
    mlir::Value result;

    if (rank == 2) {
        auto matmulOp = builder.create<mlir::linalg::MatmulOp>(
            loc, mlir::TypeRange(resultType),
            mlir::ValueRange({lhs, rhs}),
            mlir::ValueRange(empty));
        result = matmulOp.getResult(0);
    }
    else if (rank == 3) {
        auto batchMatmulOp = builder.create<mlir::linalg::BatchMatmulOp>(
            loc, mlir::TypeRange(resultType),
            mlir::ValueRange({lhs, rhs}),
            mlir::ValueRange(empty));
        result = batchMatmulOp.getResult(0);
    }
    else {
        PLOG_DEBUG << "Unsupported rank for MatMul: " << rank;
        return nullptr;
    }

    PLOG_DEBUG << "[MatMulEmitter] Saving output '" << outputNames[0] << "' to tensorMap\n";
    tensorMap[outputNames[0]] = result;

    return result;
}

} // namespace mlir_gen
} // namespace tcc
