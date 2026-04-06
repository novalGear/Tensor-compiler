// tests/test_ReLU_emitter.cpp
#include <gtest/gtest.h>
#include "graph.hpp"
#include "MiddleEnd/MLIR/MLIRGenerator.hpp"

using namespace tcc;
using namespace tcc::mlir_gen;

class ReLUEmitterTest : public ::testing::Test {
protected:
    ComputeGraph createReLUGraph(const std::vector<size_t>& dims) {
        ComputeGraph graph;

        TensorID input = "input", out = "output";

        graph.tensor_map[input] = {dims, NO_PRODUCER, {}, true, false};
        graph.tensor_map[out] = {dims, 0, {}, false, false};

        ReLUNode ReLUNode;
        ReLUNode.name = "ReLU1";
        ReLUNode.input_tensors = {input};
        ReLUNode.output_tensors = {out};

        graph.nodes.push_back(ReLUNode);
        graph.tensor_map[out].producer_node_id = 0;
        graph.tensor_map[input].consumer_node_ids = {0};

        return graph;
    }
};

TEST_F(ReLUEmitterTest, ReLU2DMatrix) {
    auto graph = createReLUGraph({2, 3});

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}

TEST_F(ReLUEmitterTest, ReLU3DTensor) {
    auto graph = createReLUGraph({2, 3, 4});

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}

TEST_F(ReLUEmitterTest, ReLUScalar) {
    auto graph = createReLUGraph({});

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}
