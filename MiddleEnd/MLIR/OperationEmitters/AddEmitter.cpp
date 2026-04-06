#include "AddEmitter.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"

#include <iostream>

namespace tcc {
namespace mlir_gen {

mlir::Value AddEmitter::emit(const std::vector<mlir::Value>& inputs,
                              const std::vector<std::string>& outputNames,
                              const std::vector<size_t>& outputDims) {

    std::cout << "[DEBUG] AddEmitter: creating add for " << outputNames[0] << "\n";
    std::cout << "[DEBUG]   inputs size: " << inputs.size() << "\n";
    auto loc = mlir::UnknownLoc::get(builder.getContext());

    if (inputs.size() != 2) {
        llvm::errs() << "Add operation requires exactly 2 inputs\n";
        return nullptr;
    }

    auto lhsType = inputs[0].getType().cast<mlir::RankedTensorType>();
    auto rhsType = inputs[1].getType().cast<mlir::RankedTensorType>();
    auto elementType = lhsType.getElementType();

    // Создание типа результата
    llvm::SmallVector<int64_t> mlirDims;
    for (auto dim : outputDims) {
        mlirDims.push_back(static_cast<int64_t>(dim));
    }
    auto resultType = mlir::RankedTensorType::get(mlirDims, elementType);

    // Пустой тензор для результата
    auto empty = builder.create<mlir::tensor::EmptyOp>(loc, mlirDims, elementType);

    // Индексные мапы (identity) для 2 входов + 1 выхода
    int rank = resultType.getRank();
    auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(rank, builder.getContext());

    // Создаем ArrayAttr из мап
    llvm::SmallVector<mlir::Attribute> mapAttrs;
    for (int i = 0; i < 3; ++i) {  // 3 операнда: lhs, rhs, output
        mapAttrs.push_back(mlir::AffineMapAttr::get(identityMap));
    }
    auto indexingMapsAttr = mlir::ArrayAttr::get(builder.getContext(), mapAttrs);

    // Типы итераторов
    llvm::SmallVector<mlir::Attribute> iterAttrs;
    for (int i = 0; i < rank; ++i) {
        iterAttrs.push_back(mlir::linalg::IteratorTypeAttr::get(
            builder.getContext(), mlir::utils::IteratorType::parallel));
    }
    auto iteratorTypesAttr = mlir::ArrayAttr::get(builder.getContext(), iterAttrs);

    // Создание linalg.generic
    auto generic = builder.create<mlir::linalg::GenericOp>(
        loc,
        mlir::TypeRange(resultType),
        mlir::ValueRange(inputs),
        mlir::ValueRange(empty),
        indexingMapsAttr,
        iteratorTypesAttr,
        /*doc=*/nullptr,
        /*library_call=*/nullptr);

    // Создание тела операции
    mlir::Region& region = generic.getRegion();
    region.push_back(new mlir::Block);
    mlir::Block& block = region.front();

    block.addArguments({elementType, elementType, elementType}, {loc, loc, loc});

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&block);

    auto arg0 = block.getArgument(0);
    auto arg1 = block.getArgument(1);
    auto sum = builder.create<mlir::arith::AddFOp>(loc, arg0, arg1);

    builder.create<mlir::linalg::YieldOp>(loc, mlir::ValueRange(sum));

    auto result = generic.getResult(0);
    tensorMap[outputNames[0]] = result;

    std::cout << "[DEBUG] AddEmitter: saved " << outputNames[0] << " to tensorMap\n";

    return result;
}

} // namespace mlir_gen
} // namespace tcc
