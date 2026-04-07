// tests/test_Relu_emitter.cpp
#include <gtest/gtest.h>
#include "graph.hpp"
#include "MiddleEnd/MLIR/MLIRGenerator.hpp"

using namespace tcc;
using namespace tcc::mlir_gen;

class ReluEmitterTest : public ::testing::Test {
protected:
    ComputeGraph createReluGraph(const std::vector<size_t>& dims) {
        ComputeGraph graph;

        TensorID input = "input", out = "output";

        graph.tensor_descr_map[input] = {dims, NO_PRODUCER, {}, true, false};
        graph.tensor_descr_map[out] = {dims, 0, {}, false, false};

        ReluNode ReluNode;
        ReluNode.name = "Relu1";
        ReluNode.input_tensors = {input};
        ReluNode.output_tensors = {out};

        graph.nodes.push_back(ReluNode);
        graph.tensor_descr_map[out].producer_node_id = 0;
        graph.tensor_descr_map[input].consumer_node_ids = {0};

        return graph;
    }
};

TEST_F(ReluEmitterTest, Relu2DMatrix) {
    auto graph = createReluGraph({2, 3});

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}

TEST_F(ReluEmitterTest, Relu3DTensor) {
    auto graph = createReluGraph({2, 3, 4});

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}

TEST_F(ReluEmitterTest, ReluScalar) {
    auto graph = createReluGraph({});

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}
