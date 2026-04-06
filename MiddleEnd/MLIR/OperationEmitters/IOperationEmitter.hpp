// IOperationEmitter.hpp
#pragma once

#include "mlir/IR/Value.h"
#include <string>
#include <vector>

namespace tcc {
namespace mlir_gen {

class IOperationEmitter {
public:
    virtual ~IOperationEmitter() = default;

    // Основной метод эмиттера
    virtual mlir::Value emit(const std::vector<mlir::Value>& inputs,
                             const std::vector<std::string>& outputNames,
                             const std::vector<size_t>& outputDims) = 0;

    // Специальный метод для констант (с значениями)
    virtual mlir::Value emitConstant(const std::vector<float>& values,
                                      const std::vector<std::string>& outputNames,
                                      const std::vector<size_t>& outputDims) {
        // Базовая реализация — вызвать обычный emit (будет переопределено в ConstantEmitter)
        return emit({}, outputNames, outputDims);
    }
};

} // namespace mlir_gen
} // namespace tcc
