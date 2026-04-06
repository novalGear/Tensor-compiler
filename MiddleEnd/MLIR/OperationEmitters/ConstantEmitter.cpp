// ConstantEmitter.cpp
#include "ConstantEmitter.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <iostream>

namespace tcc {
namespace mlir_gen {

mlir::Value ConstantEmitter::emit(const std::vector<mlir::Value>& inputs,
                                   const std::vector<std::string>& outputNames,
                                   const std::vector<size_t>& outputDims) {
    // Заглушка: создаем тензор из единиц
    std::vector<float> defaultValues;
    size_t numElements = 1;
    for (auto dim : outputDims) numElements *= dim;
    defaultValues.assign(numElements, 1.0f);

    return emitConstant(defaultValues, outputNames, outputDims);
}

mlir::Value ConstantEmitter::emitConstant(const std::vector<float>& values,
                                           const std::vector<std::string>& outputNames,
                                           const std::vector<size_t>& outputDims) {
    std::cout << "[DEBUG] ConstantEmitter: creating constant for " << outputNames[0] << "\n";

    auto loc = mlir::UnknownLoc::get(builder.getContext());

    llvm::SmallVector<int64_t> mlirDims;
    for (auto dim : outputDims) {
        mlirDims.push_back(static_cast<int64_t>(dim));
    }

    auto elementType = mlir::Float32Type::get(builder.getContext());
    auto tensorType = mlir::RankedTensorType::get(mlirDims, elementType);

    // Создаем DenseElementsAttr
    std::vector<float> actualValues = values;
    size_t numElements = 1;
    for (auto dim : outputDims) numElements *= dim;
    if (actualValues.size() != numElements) {
        actualValues.assign(numElements, 0.0f);
    }

    auto denseAttr = mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef(actualValues));
    auto constantOp = builder.create<mlir::arith::ConstantOp>(loc, tensorType, denseAttr);

    // СОХРАНЯЕМ В tensorMap
    tensorMap[outputNames[0]] = constantOp.getResult();
    std::cout << "[DEBUG] ConstantEmitter: saved " << outputNames[0] << " to tensorMap\n";

    return constantOp.getResult();
}

mlir::Value ConstantEmitter::createConstantTensor(const std::vector<float>& values,
                                                    const std::vector<size_t>& dims) {
    auto loc = mlir::UnknownLoc::get(builder.getContext());

    llvm::SmallVector<int64_t> mlirDims;
    for (auto dim : dims) {
        mlirDims.push_back(static_cast<int64_t>(dim));
    }

    size_t numElements = 1;
    for (auto dim : dims) {
        numElements *= dim;
    }

    // Если values пустые или не соответствуют размеру, заполняем нулями
    std::vector<float> actualValues = values;
    if (actualValues.size() != numElements) {
        actualValues.assign(numElements, 0.0f);
    }

    auto elementType = mlir::Float32Type::get(builder.getContext());
    auto tensorType = mlir::RankedTensorType::get(mlirDims, elementType);

    llvm::ArrayRef<float> valueRef(actualValues);
    auto denseAttr = mlir::DenseElementsAttr::get(tensorType, valueRef);

    auto constantOp = builder.create<mlir::arith::ConstantOp>(loc, tensorType, denseAttr);

    return constantOp.getResult();
}

mlir::Value ConstantEmitter::createConstantScalar(float value) {
    auto loc = mlir::UnknownLoc::get(builder.getContext());

    auto floatType = mlir::Float32Type::get(builder.getContext());
    auto attr = mlir::FloatAttr::get(floatType, value);

    auto constantOp = builder.create<mlir::arith::ConstantOp>(loc, floatType, attr);

    return constantOp.getResult();
}

} // namespace mlir_gen
} // namespace tcc
