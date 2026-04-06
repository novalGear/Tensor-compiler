// tests/test_type_converter.cpp
#include <gtest/gtest.h>
#include "MiddleEnd/MLIR/TypeConverter.hpp"
#include "mlir/IR/MLIRContext.h"

using namespace tcc::mlir_gen;

class TypeConverterTest : public ::testing::Test {
protected:
    void SetUp() override {
        context = std::make_unique<mlir::MLIRContext>();
        converter = std::make_unique<TypeConverter>(context.get());
    }

    std::unique_ptr<mlir::MLIRContext> context;
    std::unique_ptr<TypeConverter> converter;
};

TEST_F(TypeConverterTest, StaticTensor2D) {
    auto type = converter->toTensorType({2, 3});
    EXPECT_TRUE(type.hasStaticShape());
    EXPECT_EQ(type.getRank(), 2);
    EXPECT_EQ(type.getDimSize(0), 2);
    EXPECT_EQ(type.getDimSize(1), 3);
}

TEST_F(TypeConverterTest, GetNumElements) {
    size_t num = TypeConverter::getNumElements({2, 3, 4});
    EXPECT_EQ(num, 24);
}

TEST_F(TypeConverterTest, IsDynamicDim) {
    // Исправлено: проверяем, что -1 это динамическая размерность
    EXPECT_TRUE(TypeConverter::isDynamicDim(-1));
    EXPECT_FALSE(TypeConverter::isDynamicDim(5));
    EXPECT_FALSE(TypeConverter::isDynamicDim(0));
}

TEST_F(TypeConverterTest, ToMLIRDim) {
    EXPECT_EQ(TypeConverter::toMLIRDim(5), 5);
    EXPECT_EQ(TypeConverter::toMLIRDim(0), 0);
}
