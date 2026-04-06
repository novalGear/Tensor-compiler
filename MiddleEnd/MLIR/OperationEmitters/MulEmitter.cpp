#include "MulEmitter.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"

namespace tcc {
namespace mlir_gen {

mlir::Value MulEmitter::emit(const std::vector<mlir::Value>& inputs,
                              const std::vector<std::string>& outputNames,
                              const std::vector<size_t>& outputDims) {

    auto loc = mlir::UnknownLoc::get(builder.getContext());

    if (inputs.size() != 2) {
        llvm::errs() << "Mul operation requires exactly 2 inputs\n";
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
    llvm::SmallVector<mlir::AffineMap> indexingMaps = {identityMap, identityMap, identityMap};

    // Конвертируем в ArrayAttr
    llvm::SmallVector<mlir::Attribute> mapAttrs;
    for (auto& map : indexingMaps) {
        mapAttrs.push_back(mlir::AffineMapAttr::get(map));
    }
    auto indexingMapsArrayAttr = mlir::ArrayAttr::get(builder.getContext(), mapAttrs);

    // Типы итераторов (все parallel)
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        rank, mlir::utils::IteratorType::parallel);

    // Конвертируем iteratorTypes в ArrayAttr
    llvm::SmallVector<mlir::Attribute> iterAttrs;
    for (auto type : iteratorTypes) {
        iterAttrs.push_back(mlir::linalg::IteratorTypeAttr::get(builder.getContext(), type));
    }
    auto iteratorTypesArrayAttr = mlir::ArrayAttr::get(builder.getContext(), iterAttrs);

    // Создание linalg.generic
    auto generic = builder.create<mlir::linalg::GenericOp>(
        loc,
        resultType,
        inputs,
        mlir::ValueRange(empty),
        indexingMapsArrayAttr,
        iteratorTypesArrayAttr,
        /*doc=*/nullptr,
        /*library_call=*/nullptr);

    // Создание тела операции
    mlir::Region& region = generic.getRegion();
    region.push_back(new mlir::Block);
    mlir::Block& block = region.front();

    // Добавление аргументов блока
    block.addArguments({elementType, elementType, elementType},
                       {loc, loc, loc});

    // Установка точки вставки внутрь блока
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&block);

    // Тело: умножение
    auto arg0 = block.getArgument(0);
    auto arg1 = block.getArgument(1);
    auto product = builder.create<mlir::arith::MulFOp>(loc, arg0, arg1);

    builder.create<mlir::linalg::YieldOp>(loc, mlir::ValueRange(product));

    // Сохранение результата
    auto result = generic.getResult(0);
    tensorMap[outputNames[0]] = result;

    return result;
}

} // namespace mlir_gen
} // namespace tcc
