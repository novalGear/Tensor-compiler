// ConstantEmitter.cpp
#include "ConstantEmitter.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace tcc {
namespace mlir_gen {

mlir::Value ConstantEmitter::emit(const std::vector<mlir::Value>& inputs,
                                   const std::vector<std::string>& outputNames,
                                   const std::vector<size_t>& outputDims) {

    auto loc = mlir::UnknownLoc::get(builder.getContext());

    // Константа не имеет входных тензоров (inputs пуст)
    // Входные данные хранятся в самой структуре ConstantNode,
    // но здесь мы их получаем через другой механизм.
    // Для простоты предположим, что значения передаются через outputDims
    // или отдельный параметр. В реальности нужно расширить интерфейс.

    // ВАЖНО: В текущей реализации IOperationEmitter не передает значения константы.
    // Нужно либо расширить интерфейс, либо получать значения из другого места.
    // Ниже показан пример, когда значения передаются как дополнительный параметр.

    // Создаем константный тензор
    // Для примера создадим тензор 2x3 с единицами
    std::vector<float> exampleValues = {
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f
    };

    mlir::Value result;

    if (outputDims.empty()) {
        // Скалярная константа
        result = createConstantScalar(1.0f);
    } else {
        // Тензорная константа
        result = createConstantTensor(exampleValues, outputDims);
    }

    // Сохранение результата
    tensorMap[outputNames[0]] = result;

    return result;
}

mlir::Value ConstantEmitter::createConstantTensor(const std::vector<float>& values,
                                                    const std::vector<size_t>& dims) {
    auto loc = mlir::UnknownLoc::get(builder.getContext());

    // Преобразование размерностей в int64_t
    llvm::SmallVector<int64_t> mlirDims;
    for (auto dim : dims) {
        mlirDims.push_back(static_cast<int64_t>(dim));
    }

    // Вычисляем общее количество элементов
    size_t numElements = 1;
    for (auto dim : dims) {
        numElements *= dim;
    }

    assert(values.size() == numElements && "Number of values doesn't match tensor size");

    // Создаем DenseElementsAttr (атрибут MLIR для хранения константных данных)
    auto elementType = mlir::Float32Type::get(builder.getContext());
    auto tensorType = mlir::RankedTensorType::get(mlirDims, elementType);

    // Преобразуем vector<float> в llvm::ArrayRef<float>
    llvm::ArrayRef<float> valueRef(values);

    // Создаем атрибут из значений
    auto denseAttr = mlir::DenseElementsAttr::get(tensorType, valueRef);

    // Создаем константную операцию
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
