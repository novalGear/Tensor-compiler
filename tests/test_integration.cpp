// tests/test_integration.cpp
#include <gtest/gtest.h>
#include "graph.hpp"
#include "MiddleEnd/MLIR/MLIRGenerator.hpp"

using namespace tcc;
using namespace tcc::mlir_gen;

class IntegrationTest : public ::testing::Test {
protected:
    // Модель: (x + y) * z
    ComputeGraph createAddMulGraph() {
        ComputeGraph graph;

        TensorID x = "x", y = "y", z = "z";
        TensorID sum = "sum";
        TensorID result = "result";

        // Входы
        graph.tensor_map[x] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[y] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[z] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[sum] = {{2, 3}, 0, {}, false, false};
        graph.tensor_map[result] = {{2, 3}, 1, {}, false, false};

        // Add
        AddNode addNode;
        addNode.name = "add";
        addNode.input_tensors = {x, y};
        addNode.output_tensors = {sum};

        // Mul
        MulNode mulNode;
        mulNode.name = "mul";
        mulNode.input_tensors = {sum, z};
        mulNode.output_tensors = {result};

        graph.nodes.push_back(addNode);
        graph.nodes.push_back(mulNode);

        graph.tensor_map[sum].producer_node_id = 0;
        graph.tensor_map[result].producer_node_id = 1;
        graph.tensor_map[x].consumer_node_ids = {0};
        graph.tensor_map[y].consumer_node_ids = {0};
        graph.tensor_map[z].consumer_node_ids = {1};
        graph.tensor_map[sum].consumer_node_ids = {1};

        return graph;
    }

    // Модель: ReLU(A + B)
    ComputeGraph createAddReLUGraph() {
        ComputeGraph graph;

        TensorID A = "A", B = "B";
        TensorID sum = "sum";
        TensorID result = "result";

        graph.tensor_map[A] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[B] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[sum] = {{2, 3}, 0, {}, false, false};
        graph.tensor_map[result] = {{2, 3}, 1, {}, false, false};

        AddNode addNode;
        addNode.name = "add";
        addNode.input_tensors = {A, B};
        addNode.output_tensors = {sum};

        ReLUNode ReLUNode;
        ReLUNode.name = "ReLU";
        ReLUNode.input_tensors = {sum};
        ReLUNode.output_tensors = {result};

        graph.nodes.push_back(addNode);
        graph.nodes.push_back(ReLUNode);

        graph.tensor_map[sum].producer_node_id = 0;
        graph.tensor_map[result].producer_node_id = 1;
        graph.tensor_map[A].consumer_node_ids = {0};
        graph.tensor_map[B].consumer_node_ids = {0};
        graph.tensor_map[sum].consumer_node_ids = {1};

        return graph;
    }
};

TEST_F(IntegrationTest, AddThenMul) {
    auto graph = createAddMulGraph();

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}

TEST_F(IntegrationTest, AddThenReLU) {
    auto graph = createAddReLUGraph();

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}
