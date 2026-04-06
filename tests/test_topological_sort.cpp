// tests/test_topological_sort.cpp
#include <gtest/gtest.h>
#include "graph.hpp"
#include "MiddleEnd/MLIR/MLIRGenerator.hpp"

using namespace tcc;

class TopologicalSortTest : public ::testing::Test {
protected:
    ComputeGraph createTwoNodeGraph() {
        ComputeGraph graph;

        TensorID a = "x", b = "y", c = "mid", d = "out";

        graph.tensor_map[a] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[b] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[c] = {{2, 3}, 0, {}, false, false};
        graph.tensor_map[d] = {{2, 3}, 1, {}, false, false};

        AddNode addNode;
        addNode.name = "add1";
        addNode.input_tensors = {a, b};
        addNode.output_tensors = {c};

        MulNode mulNode;
        mulNode.name = "mul1";
        mulNode.input_tensors = {c, a};
        mulNode.output_tensors = {d};

        graph.nodes.push_back(addNode);
        graph.nodes.push_back(mulNode);

        graph.tensor_map[c].producer_node_id = 0;
        graph.tensor_map[d].producer_node_id = 1;
        graph.tensor_map[a].consumer_node_ids = {0, 1};
        graph.tensor_map[b].consumer_node_ids = {0};
        graph.tensor_map[c].consumer_node_ids = {1};

        return graph;
    }
};

TEST_F(TopologicalSortTest, TwoNodeChain) {
    auto graph = createTwoNodeGraph();
    EXPECT_EQ(graph.nodes.size(), 2);

    auto& addTensor = graph.tensor_map["mid"];
    EXPECT_EQ(addTensor.producer_node_id, 0);
}
