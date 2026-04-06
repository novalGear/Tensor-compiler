// tests/test_graph_methods.cpp
#include <gtest/gtest.h>
#include "graph.hpp"

using namespace tcc;

class GraphMethodsTest : public ::testing::Test {
protected:
    ComputeGraph createTestGraph() {
        ComputeGraph graph;

        TensorID input1 = "input1", input2 = "input2";
        TensorID intermediate = "mid";
        TensorID output = "output";

        // Входы
        graph.tensor_map[input1] = {{2, 3}, NO_PRODUCER, {0}, true, false};
        graph.tensor_map[input2] = {{2, 3}, NO_PRODUCER, {0}, true, false};

        // Промежуточный
        graph.tensor_map[intermediate] = {{2, 3}, 0, {1}, false, false};

        // Выход
        graph.tensor_map[output] = {{2, 3}, 1, {}, false, false};

        AddNode add;
        add.name = "add";
        add.input_tensors = {input1, input2};
        add.output_tensors = {intermediate};

        MulNode mul;
        mul.name = "mul";
        mul.input_tensors = {intermediate, input1};
        mul.output_tensors = {output};

        graph.nodes.push_back(add);
        graph.nodes.push_back(mul);

        graph.tensor_map[intermediate].producer_node_id = 0;
        graph.tensor_map[output].producer_node_id = 1;
        graph.tensor_map[input1].consumer_node_ids = {0, 1};
        graph.tensor_map[input2].consumer_node_ids = {0};
        graph.tensor_map[intermediate].consumer_node_ids = {1};

        return graph;
    }
};

TEST_F(GraphMethodsTest, CollectInputs) {
    auto graph = createTestGraph();
    auto inputs = graph.collectInputs();

    EXPECT_EQ(inputs.size(), 2);
    EXPECT_TRUE(std::find(inputs.begin(), inputs.end(), "input1") != inputs.end());
    EXPECT_TRUE(std::find(inputs.begin(), inputs.end(), "input2") != inputs.end());
}

TEST_F(GraphMethodsTest, CollectOutputs) {
    auto graph = createTestGraph();
    auto outputs = graph.collectOutputs();

    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0], "output");
}

TEST_F(GraphMethodsTest, GetTensorDims) {
    auto graph = createTestGraph();
    auto dims = graph.getTensorDims("input1");

    EXPECT_EQ(dims.size(), 2);
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 3);
}

TEST_F(GraphMethodsTest, GetNonExistentTensorDims) {
    auto graph = createTestGraph();
    auto dims = graph.getTensorDims("nonexistent");

    EXPECT_TRUE(dims.empty());
}
