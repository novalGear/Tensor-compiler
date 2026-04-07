// TypeConverter.cpp
#include "TypeConverter.hpp"
#include <iostream>

#include "plog/Log.h"

namespace tcc {
namespace mlir_gen {

TypeConverter::TypeConverter(mlir::MLIRContext* ctx)
    : context(ctx), elementType(mlir::Float32Type::get(ctx)) {}

mlir::RankedTensorType TypeConverter::toTensorType(const std::vector<size_t>& dims) {
    llvm::SmallVector<int64_t> mlirDims;
    for (auto dim : dims) {
        mlirDims.push_back(static_cast<int64_t>(dim));
    }
    return mlir::RankedTensorType::get(mlirDims, elementType);
}

mlir::Type TypeConverter::getElementType() const {
    return elementType;
}

bool TypeConverter::isDynamicDim(int64_t dim) {
    PLOG_DEBUG << "isDynamicDim called with dim=" << dim
              << ", kDynamic=" << mlir::ShapedType::kDynamic << "\n";
    return dim == mlir::ShapedType::kDynamic;
}

int64_t TypeConverter::toMLIRDim(size_t dim) {
    return static_cast<int64_t>(dim);
}

size_t TypeConverter::getNumElements(const std::vector<size_t>& dims) {
    size_t num = 1;
    for (auto dim : dims) {
        num *= dim;
    }
    return num;
}

} // namespace mlir_gen
} // namespace tcc
