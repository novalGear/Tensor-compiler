// tests/test_mul_emitter.cpp
#include <gtest/gtest.h>
#include "graph.hpp"
#include "MiddleEnd/MLIR/MLIRGenerator.hpp"

using namespace tcc;
using namespace tcc::mlir_gen;

class MulEmitterTest : public ::testing::Test {
protected:
    ComputeGraph createMulGraph(const std::vector<size_t>& dims) {
        ComputeGraph graph;

        TensorID a = "input_a", b = "input_b", out = "output";

        graph.tensor_descr_map[a] = {dims, NO_PRODUCER, {}, true, false};
        graph.tensor_descr_map[b] = {dims, NO_PRODUCER, {}, true, false};
        graph.tensor_descr_map[out] = {dims, 0, {}, false, false};

        MulNode mulNode;
        mulNode.name = "mul1";
        mulNode.input_tensors = {a, b};
        mulNode.output_tensors = {out};

        graph.nodes.push_back(mulNode);
        graph.tensor_descr_map[out].producer_node_id = 0;
        graph.tensor_descr_map[a].consumer_node_ids = {0};
        graph.tensor_descr_map[b].consumer_node_ids = {0};

        return graph;
    }
};

TEST_F(MulEmitterTest, Mul2DMatrix) {
    auto graph = createMulGraph({2, 3});

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}

TEST_F(MulEmitterTest, Mul3DTensor) {
    auto graph = createMulGraph({2, 3, 4});

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}
