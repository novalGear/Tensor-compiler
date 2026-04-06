// ReLUEmitter.cpp
#include "ReLUEmitter.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"

namespace tcc {
namespace mlir_gen {

mlir::Value ReLUEmitter::emit(const std::vector<mlir::Value>& inputs,
                               const std::vector<std::string>& outputNames,
                               const std::vector<size_t>& outputDims) {

    auto loc = mlir::UnknownLoc::get(builder.getContext());

    if (inputs.size() != 1) {
        llvm::errs() << "ReLU operation requires exactly 1 input\n";
        return nullptr;
    }

    auto inputType = inputs[0].getType().cast<mlir::RankedTensorType>();
    auto elementType = inputType.getElementType();

    // Создание типа результата
    llvm::SmallVector<int64_t> mlirDims;
    for (auto dim : outputDims) {
        mlirDims.push_back(static_cast<int64_t>(dim));
    }
    auto resultType = mlir::RankedTensorType::get(mlirDims, elementType);

    // Пустой тензор для результата
    auto empty = builder.create<mlir::tensor::EmptyOp>(loc, mlirDims, elementType);

    // Индексные мапы
    int rank = resultType.getRank();
    auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(rank, builder.getContext());

    llvm::SmallVector<mlir::Attribute> mapAttrs;
    for (int i = 0; i < 2; ++i) {
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
        nullptr, nullptr);

    // Тело операции
    mlir::Region& region = generic.getRegion();
    region.push_back(new mlir::Block);
    mlir::Block& block = region.front();

    block.addArguments({elementType, elementType}, {loc, loc});

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&block);

    auto arg0 = block.getArgument(0);

    // Создаем константу 0
    auto zero = builder.create<mlir::arith::ConstantOp>(
        loc, elementType, builder.getZeroAttr(elementType));

    // Вариант 1: Используем MaxFOp (если есть)
    auto max = builder.create<mlir::arith::MaxFOp>(loc, arg0, zero);

    // Вариант 2: Если MaxFOp не работает, используем cmp + select (закомментирован)
    // auto cmp = builder.create<mlir::arith::CmpFOp>(
    //     loc, mlir::arith::CmpFPredicate::UGT, arg0, zero);
    // auto max = builder.create<mlir::arith::SelectOp>(loc, cmp, arg0, zero);

    builder.create<mlir::linalg::YieldOp>(loc, mlir::ValueRange(max));

    auto result = generic.getResult(0);
    tensorMap[outputNames[0]] = result;

    return result;
}

} // namespace mlir_gen
} // namespace tcc
