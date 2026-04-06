// ConstantEmitter.hpp
#pragma once

#include "IOperationEmitter.hpp"
#include "mlir/IR/Builders.h"
#include <unordered_map>
#include <vector>

namespace tcc {
namespace mlir_gen {

class ConstantEmitter : public IOperationEmitter {
public:
    ConstantEmitter(mlir::OpBuilder& builder,
                    std::unordered_map<std::string, mlir::Value>& tensorMap)
        : builder(builder), tensorMap(tensorMap) {}

    mlir::Value emit(const std::vector<mlir::Value>& inputs,
                     const std::vector<std::string>& outputNames,
                     const std::vector<size_t>& outputDims) override;

    // Переопределяем метод для констант с значениями
    mlir::Value emitConstant(const std::vector<float>& values,
                              const std::vector<std::string>& outputNames,
                              const std::vector<size_t>& outputDims) override;

private:
    mlir::OpBuilder& builder;
    std::unordered_map<std::string, mlir::Value>& tensorMap;

    mlir::Value createConstantTensor(const std::vector<float>& values,
                                      const std::vector<size_t>& dims);
    mlir::Value createConstantScalar(float value);
};

} // namespace mlir_gen
} // namespace tcc
