// tests/test_topological_sort.cpp
#include <gtest/gtest.h>
#include "graph.hpp"

using namespace tcc;

class TopologicalSortTest : public ::testing::Test {
protected:
    // Граф: x, y -> Add -> mid -> Mul -> out
    ComputeGraph createLinearGraph() {
        ComputeGraph graph;

        TensorID x = "x", y = "y", mid = "mid", out = "out";

        // Тензоры
        graph.tensor_map[x] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[y] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[mid] = {{2, 3}, 0, {1}, false, false};
        graph.tensor_map[out] = {{2, 3}, 1, {}, false, false};

        // Узел 0: Add
        AddNode add;
        add.name = "add";
        add.input_tensors = {x, y};
        add.output_tensors = {mid};

        // Узел 1: Mul
        MulNode mul;
        mul.name = "mul";
        mul.input_tensors = {mid, x};
        mul.output_tensors = {out};

        graph.nodes.push_back(add);
        graph.nodes.push_back(mul);

        // Связи
        graph.tensor_map[mid].producer_node_id = 0;
        graph.tensor_map[out].producer_node_id = 1;
        graph.tensor_map[x].consumer_node_ids = {0, 1};
        graph.tensor_map[y].consumer_node_ids = {0};
        graph.tensor_map[mid].consumer_node_ids = {1};

        return graph;
    }

    // Независимые узлы: Add1 и Add2
    ComputeGraph createIndependentGraph() {
        ComputeGraph graph;

        TensorID a = "a", b = "b", out1 = "out1";
        TensorID c = "c", d = "d", out2 = "out2";

        graph.tensor_map[a] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[b] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[out1] = {{2, 3}, 0, {}, false, false};
        graph.tensor_map[c] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[d] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[out2] = {{2, 3}, 1, {}, false, false};

        AddNode add1;
        add1.name = "add1";
        add1.input_tensors = {a, b};
        add1.output_tensors = {out1};

        AddNode add2;
        add2.name = "add2";
        add2.input_tensors = {c, d};
        add2.output_tensors = {out2};

        graph.nodes.push_back(add1);
        graph.nodes.push_back(add2);

        graph.tensor_map[out1].producer_node_id = 0;
        graph.tensor_map[out2].producer_node_id = 1;
        graph.tensor_map[a].consumer_node_ids = {0};
        graph.tensor_map[b].consumer_node_ids = {0};
        graph.tensor_map[c].consumer_node_ids = {1};
        graph.tensor_map[d].consumer_node_ids = {1};

        return graph;
    }

    // Один узел без зависимостей
    ComputeGraph createSingleNodeGraph() {
        ComputeGraph graph;

        TensorID input = "input", out = "out";

        graph.tensor_map[input] = {{2, 3}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[out] = {{2, 3}, 0, {}, false, false};

        ReLUNode ReLU;
        ReLU.name = "ReLU";
        ReLU.input_tensors = {input};
        ReLU.output_tensors = {out};

        graph.nodes.push_back(ReLU);
        graph.tensor_map[out].producer_node_id = 0;
        graph.tensor_map[input].consumer_node_ids = {0};

        return graph;
    }
};

// ================================================================
// Тест 1: Линейный граф (Add -> Mul)
// ================================================================
TEST_F(TopologicalSortTest, LinearGraph) {
    auto graph = createLinearGraph();
    auto order = graph.topologicalSort();

    EXPECT_EQ(order.size(), 2);
    // Add (узел 0) должен быть перед Mul (узел 1)
    EXPECT_EQ(order[0], 0);
    EXPECT_EQ(order[1], 1);
}

// ================================================================
// Тест 2: Независимые узлы
// ================================================================
TEST_F(TopologicalSortTest, IndependentGraph) {
    auto graph = createIndependentGraph();
    auto order = graph.topologicalSort();

    EXPECT_EQ(order.size(), 2);
    // Оба узла независимы, порядок может быть любым
    // Просто проверяем, что оба присутствуют
    EXPECT_TRUE((order[0] == 0 && order[1] == 1) ||
                (order[0] == 1 && order[1] == 0));
}

// ================================================================
// Тест 3: Один узел
// ================================================================
TEST_F(TopologicalSortTest, SingleNodeGraph) {
    auto graph = createSingleNodeGraph();
    auto order = graph.topologicalSort();

    EXPECT_EQ(order.size(), 1);
    EXPECT_EQ(order[0], 0);
}

// ================================================================
// Тест 4: Пустой граф
// ================================================================
TEST_F(TopologicalSortTest, EmptyGraph) {
    ComputeGraph graph;
    auto order = graph.topologicalSort();

    EXPECT_TRUE(order.empty());
}
