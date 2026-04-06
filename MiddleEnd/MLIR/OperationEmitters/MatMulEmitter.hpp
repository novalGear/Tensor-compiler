// MatMulEmitter.hpp
#pragma once

#include "IOperationEmitter.hpp"
#include "mlir/IR/Builders.h"
#include <unordered_map>

namespace tcc {
namespace mlir_gen {

class MatMulEmitter : public IOperationEmitter {
public:
    MatMulEmitter(mlir::OpBuilder& builder,
                  std::unordered_map<std::string, mlir::Value>& tensorMap)
        : builder(builder), tensorMap(tensorMap) {}

    mlir::Value emit(const std::vector<mlir::Value>& inputs,
                     const std::vector<std::string>& outputNames,
                     const std::vector<size_t>& outputDims) override;

private:
    mlir::OpBuilder& builder;
    std::unordered_map<std::string, mlir::Value>& tensorMap;

    // Вспомогательные методы
    int getRank(const mlir::Value& tensor);
    mlir::Value broadcastToRank(mlir::Value tensor, int targetRank);
    bool isBatchMatMul(const std::vector<size_t>& lhsDims,
                       const std::vector<size_t>& rhsDims);
};

} // namespace mlir_gen
} // namespace tcc
