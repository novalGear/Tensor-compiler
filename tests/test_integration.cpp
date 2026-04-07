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
        graph.tensor_descr_map[x] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_descr_map[y] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_descr_map[z] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_descr_map[sum] = {{2, 3}, 0, {}, false, false};
        graph.tensor_descr_map[result] = {{2, 3}, 1, {}, false, false};

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

        graph.tensor_descr_map[sum].producer_node_id = 0;
        graph.tensor_descr_map[result].producer_node_id = 1;
        graph.tensor_descr_map[x].consumer_node_ids = {0};
        graph.tensor_descr_map[y].consumer_node_ids = {0};
        graph.tensor_descr_map[z].consumer_node_ids = {1};
        graph.tensor_descr_map[sum].consumer_node_ids = {1};

        return graph;
    }

    // Модель: Relu(A + B)
    ComputeGraph createAddReluGraph() {
        ComputeGraph graph;

        TensorID A = "A", B = "B";
        TensorID sum = "sum";
        TensorID result = "result";

        graph.tensor_descr_map[A] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_descr_map[B] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_descr_map[sum] = {{2, 3}, 0, {}, false, false};
        graph.tensor_descr_map[result] = {{2, 3}, 1, {}, false, false};

        AddNode addNode;
        addNode.name = "add";
        addNode.input_tensors = {A, B};
        addNode.output_tensors = {sum};

        ReluNode ReluNode;
        ReluNode.name = "Relu";
        ReluNode.input_tensors = {sum};
        ReluNode.output_tensors = {result};

        graph.nodes.push_back(addNode);
        graph.nodes.push_back(ReluNode);

        graph.tensor_descr_map[sum].producer_node_id = 0;
        graph.tensor_descr_map[result].producer_node_id = 1;
        graph.tensor_descr_map[A].consumer_node_ids = {0};
        graph.tensor_descr_map[B].consumer_node_ids = {0};
        graph.tensor_descr_map[sum].consumer_node_ids = {1};

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

TEST_F(IntegrationTest, AddThenRelu) {
    auto graph = createAddReluGraph();

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}
