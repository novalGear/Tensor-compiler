// TypeConverter.hpp
#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include <vector>
#include <cstdint>

namespace tcc {
namespace mlir_gen {

class TypeConverter {
public:
    explicit TypeConverter(mlir::MLIRContext* context);
    ~TypeConverter() = default;

    // Преобразование размерностей в MLIR тензорный тип
    mlir::RankedTensorType toTensorType(const std::vector<size_t>& dims);

    // Получение типа элемента (по умолчанию f32)
    mlir::Type getElementType() const;

    // Проверка, является ли размерность динамической
    static bool isDynamicDim(int64_t dim);

    // Конвертация размера в MLIR формат
    static int64_t toMLIRDim(size_t dim);

    // Получение количества элементов в тензоре
    static size_t getNumElements(const std::vector<size_t>& dims);

private:
    mlir::MLIRContext* context;
    mlir::Type elementType;
};

} // namespace mlir_gen
} // namespace tcc
