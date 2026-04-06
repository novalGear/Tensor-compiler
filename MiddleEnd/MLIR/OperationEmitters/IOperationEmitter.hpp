// IOperationEmitter.h
#pragma once

#include "mlir/IR/Value.h"
#include <string>
#include <vector>

namespace tcc {
namespace mlir_gen {

// Базовый интерфейс для всех эмиттеров операций
class IOperationEmitter {
public:
    virtual ~IOperationEmitter() = default;

    // Основной метод эмиттера
    // inputs:    MLIR значения входных тензоров
    // outputNames: имена выходных тензоров (из графа)
    // outputDims: размерности выходных тензоров
    virtual mlir::Value emit(const std::vector<mlir::Value>& inputs,
                             const std::vector<std::string>& outputNames,
                             const std::vector<size_t>& outputDims) = 0;
};

} // namespace mlir_gen
} // namespace tcc
