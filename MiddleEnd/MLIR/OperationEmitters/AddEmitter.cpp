#include "AddEmitter.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"

#include "plog/Log.h"

namespace tcc {
namespace mlir_gen {

static std::string dimsToString(llvm::ArrayRef<int64_t> dims) {
    std::string result = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(dims[i]);
    }
    result += "]";
    return result;
}

static std::string affineMapToString(mlir::AffineMap map) {
    std::string result;
    llvm::raw_string_ostream os(result);
    map.print(os);
    return result;
}

mlir::Value AddEmitter::emit(const std::vector<mlir::Value>& inputs,
                              const std::vector<std::string>& outputNames,
                              const std::vector<size_t>& outputDims) {

    PLOG_DEBUG << "[DEBUG] AddEmitter: creating add for " << outputNames[0];

    auto loc = mlir::UnknownLoc::get(builder.getContext());

    if (inputs.size() != 2) {
        llvm::errs() << "[ERROR] Add operation requires exactly 2 inputs\n";
        return nullptr;
    }

    auto lhsType = inputs[0].getType().dyn_cast<mlir::RankedTensorType>();
    auto rhsType = inputs[1].getType().dyn_cast<mlir::RankedTensorType>();

    if (!lhsType || !rhsType) {
        llvm::errs() << "[ERROR] Add inputs must be ranked tensors\n";
        return nullptr;
    }

    auto elementType = lhsType.getElementType();

    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();

    // Создание типа результата
    llvm::SmallVector<int64_t> mlirDims;
    for (auto dim : outputDims) {
        mlirDims.push_back(static_cast<int64_t>(dim));
    }
    auto resultType = mlir::RankedTensorType::get(mlirDims, elementType);
    int outputRank = resultType.getRank();

    PLOG_DEBUG << "[DEBUG]   lhs shape: " << dimsToString(lhsShape) << " (rank " << lhsShape.size() << ")";
    PLOG_DEBUG << "[DEBUG]   rhs shape: " << dimsToString(rhsShape) << " (rank " << rhsShape.size() << ")";
    PLOG_DEBUG << "[DEBUG]   out shape: " << dimsToString(mlirDims) << " (rank " << outputRank << ")";

    // Пустой тензор для результата
    auto empty = builder.create<mlir::tensor::EmptyOp>(loc, mlirDims, elementType);

    // Создаем indexing maps (как ArrayRef<AffineMap>, не ArrayAttr!)
    llvm::SmallVector<mlir::AffineMap> indexingMaps;

    // 1. Map для LHS: identity map с outputRank измерениями
    //    Пример: outputRank=2 -> (d0, d1) -> (d0, d1)
    auto lhsMap = mlir::AffineMap::getMultiDimIdentityMap(outputRank, builder.getContext());
    indexingMaps.push_back(lhsMap);
    PLOG_DEBUG << "[DEBUG] LHS map: " << affineMapToString(lhsMap)
               << " (dims=" << lhsMap.getNumDims() << ", results=" << lhsMap.getNumResults() << ")";

    // 2. Map для RHS: broadcast map с выравниванием справа
    //    Пример: rhsShape=[5], outputRank=2 -> (d0, d1) -> (d1)
    //    numDims = outputRank, numResults = rank(RHS)
    llvm::SmallVector<mlir::AffineExpr> rhsExprs;
    int offset = outputRank - rhsShape.size();

    for (size_t i = 0; i < rhsShape.size(); ++i) {
        int outputDimIdx = offset + i;
        rhsExprs.push_back(mlir::getAffineDimExpr(outputDimIdx, builder.getContext()));
        PLOG_DEBUG << "[DEBUG]   rhs dim " << i << " -> output dim " << outputDimIdx;
    }
    auto rhsMap = mlir::AffineMap::get(outputRank, 0, rhsExprs, builder.getContext());
    indexingMaps.push_back(rhsMap);
    PLOG_DEBUG << "[DEBUG] RHS map: " << affineMapToString(rhsMap)
               << " (dims=" << rhsMap.getNumDims() << ", results=" << rhsMap.getNumResults() << ")";

    // 3. Map для выхода: identity map
    auto outMap = mlir::AffineMap::getMultiDimIdentityMap(outputRank, builder.getContext());
    indexingMaps.push_back(outMap);
    PLOG_DEBUG << "[DEBUG] Output map: " << affineMapToString(outMap)
               << " (dims=" << outMap.getNumDims() << ", results=" << outMap.getNumResults() << ")";

    // Создаем iterator types (как ArrayRef<IteratorType>, не ArrayAttr!)
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes;
    for (int i = 0; i < outputRank; ++i) {
        iteratorTypes.push_back(mlir::utils::IteratorType::parallel);
    }

    // Создание linalg.generic с правильными типами
    auto generic = builder.create<mlir::linalg::GenericOp>(
        loc,
        mlir::TypeRange(resultType),      // result tensor types
        mlir::ValueRange(inputs),         // inputs
        mlir::ValueRange(empty),          // outputs (init tensor)
        indexingMaps,                     // ArrayRef<AffineMap>
        iteratorTypes,                    // ArrayRef<IteratorType>
        [&](mlir::OpBuilder& nestedBuilder, mlir::Location nestedLoc, mlir::ValueRange blockArgs) {
            // blockArgs[0] - элемент из LHS
            // blockArgs[1] - элемент из RHS
            // blockArgs[2] - элемент из выхода (не используется)
            auto sum = nestedBuilder.create<mlir::arith::AddFOp>(nestedLoc, blockArgs[0], blockArgs[1]);
            nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc, mlir::ValueRange(sum));
        });

    if (!generic) {
        llvm::errs() << "[ERROR] Failed to create linalg.generic for Add\n";
        return nullptr;
    }

    auto result = generic.getResult(0);
    tensorMap[outputNames[0]] = result;

    PLOG_DEBUG << "[DEBUG] AddEmitter: successfully created add for " << outputNames[0];

    return result;
}

} // namespace mlir_gen
} // namespace tcc
