// MatMulEmitter.cpp
#include "MatMulEmitter.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace tcc {
namespace mlir_gen {

mlir::Value MatMulEmitter::emit(const std::vector<mlir::Value>& inputs,
                                 const std::vector<std::string>& outputNames,
                                 const std::vector<size_t>& outputDims) {

    auto loc = mlir::UnknownLoc::get(builder.getContext());

    if (inputs.size() != 2) {
        llvm::errs() << "MatMul operation requires exactly 2 inputs\n";
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
        // 2D матричное умножение
        auto matmulOp = builder.create<mlir::linalg::MatmulOp>(
            loc, mlir::TypeRange(resultType),
            mlir::ValueRange({lhs, rhs}),
            mlir::ValueRange(empty));
        result = matmulOp.getResult(0);
    }
    else if (rank == 3) {
        // 3D batch матричное умножение
        auto batchMatmulOp = builder.create<mlir::linalg::BatchMatmulOp>(
            loc, mlir::TypeRange(resultType),
            mlir::ValueRange({lhs, rhs}),
            mlir::ValueRange(empty));
        result = batchMatmulOp.getResult(0);
    }
    else {
        llvm::errs() << "Unsupported rank for MatMul: " << rank << "\n";
        return nullptr;
    }

    tensorMap[outputNames[0]] = result;
    return result;
}

} // namespace mlir_gen
} // namespace tcc
