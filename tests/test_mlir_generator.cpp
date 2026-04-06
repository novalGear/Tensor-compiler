// tests/test_mlir_generator.cpp
#include <gtest/gtest.h>
#include "graph.hpp"
#include "MiddleEnd/MLIR/MLIRGenerator.hpp"

using namespace tcc;
using namespace tcc::mlir_gen;

class MLIRGeneratorTest : public ::testing::Test {
protected:
    ComputeGraph createSimpleAddGraph() {
        ComputeGraph graph;

        TensorID a = "x", b = "y", c = "out";

        graph.tensor_map[a] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[b] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[c] = {{2, 3}, 0, {}, false, false};

        AddNode addNode;
        addNode.name = "add1";
        addNode.input_tensors = {a, b};
        addNode.output_tensors = {c};

        graph.nodes.push_back(addNode);
        graph.tensor_map[c].producer_node_id = 0;
        graph.tensor_map[a].consumer_node_ids.push_back(0);
        graph.tensor_map[b].consumer_node_ids.push_back(0);

        return graph;
    }

    ComputeGraph createSimpleMulGraph() {
        ComputeGraph graph;

        TensorID a = "x", b = "y", c = "out";

        graph.tensor_map[a] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[b] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[c] = {{2, 3}, 0, {}, false, false};

        MulNode mulNode;
        mulNode.name = "mul1";
        mulNode.input_tensors = {a, b};
        mulNode.output_tensors = {c};

        graph.nodes.push_back(mulNode);
        graph.tensor_map[c].producer_node_id = 0;
        graph.tensor_map[a].consumer_node_ids.push_back(0);
        graph.tensor_map[b].consumer_node_ids.push_back(0);

        return graph;
    }
};

TEST_F(MLIRGeneratorTest, GenerateAddGraph) {
    auto graph = createSimpleAddGraph();

    MLIRGenerator::Config cfg;
    std::cout << "cfg printMLIR" << std::endl;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);

    // Не проверяем результат, просто проверяем что не падает
    // Если generate вернет false, тест провалится
    EXPECT_TRUE(generator.generate(graph));
}

TEST_F(MLIRGeneratorTest, GenerateMulGraph) {
    auto graph = createSimpleMulGraph();

    MLIRGenerator::Config cfg;
    cfg.printMLIR = false;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}
