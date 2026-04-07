// tests/test_mlir_generator.cpp
#include <gtest/gtest.h>
#include "graph.hpp"
#include "MiddleEnd/MLIR/MLIRGenerator.hpp"

// Добавить необходимые includes для MLIR типов
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace tcc;
using namespace tcc::mlir_gen;

class MLIRGeneratorTest : public ::testing::Test {
protected:
    // Граф с одним входом и одной константой
    ComputeGraph createGraphWithInputAndConstant() {
        ComputeGraph graph;

        TensorID input = "input";
        TensorID const_out = "const_out";
        TensorID output = "output";

        // Входной тензор
        graph.tensor_descr_map[input] = {{2, 3}, NO_PRODUCER, {1}, true, false};

        // Константа
        graph.tensor_descr_map[const_out] = {{2, 3}, 0, {1}, false, false};

        // Выход
        graph.tensor_descr_map[output] = {{2, 3}, 1, {}, false, false};

        // Constant node (индекс 0)
        ConstantNode constNode;
        constNode.name = "const";
        constNode.output_tensors = {const_out};
        constNode.value = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

        // Add node (индекс 1)
        AddNode addNode;
        addNode.name = "add";
        addNode.input_tensors = {input, const_out};
        addNode.output_tensors = {output};

        graph.nodes.push_back(constNode);
        graph.nodes.push_back(addNode);

        // Связи
        graph.tensor_descr_map[const_out].producer_node_id = 0;
        graph.tensor_descr_map[output].producer_node_id = 1;
        graph.tensor_descr_map[input].consumer_node_ids = {1};
        graph.tensor_descr_map[const_out].consumer_node_ids = {1};

        return graph;
    }

    // Граф только с константой (нет входов)
    ComputeGraph createConstantOnlyGraph() {
        ComputeGraph graph;

        TensorID out = "out";

        graph.tensor_descr_map[out] = {{2, 2}, 0, {}, false, false};

        ConstantNode constNode;
        constNode.name = "const";
        constNode.output_tensors = {out};
        constNode.value = {1.0f, 2.0f, 3.0f, 4.0f};

        graph.nodes.push_back(constNode);
        graph.tensor_descr_map[out].producer_node_id = 0;

        return graph;
    }

    // Пустой граф
    ComputeGraph createEmptyGraph() {
        return ComputeGraph();
    }
};

// ================================================================
// Тест 1: createMainFunction должен создать функцию с аргументами
// ================================================================
TEST_F(MLIRGeneratorTest, CreateMainFunctionWithInputs) {
    auto graph = createGraphWithInputAndConstant();

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}

// ================================================================
// Тест 2: createMainFunction без входов (только константа)
// ================================================================
TEST_F(MLIRGeneratorTest, CreateMainFunctionNoInputs) {
    auto graph = createConstantOnlyGraph();

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}

// ================================================================
// Тест 3: Пустой граф
// ================================================================
TEST_F(MLIRGeneratorTest, EmptyGraph) {
    auto graph = createEmptyGraph();

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}

// ================================================================
// Тест 4: Проверка, что генерация не падает на простом Add графе
// ================================================================
TEST_F(MLIRGeneratorTest, SimpleAddGraph) {
    ComputeGraph graph;

    TensorID x = "x", y = "y", out = "out";

    graph.tensor_descr_map[x] = {{2, 3}, NO_PRODUCER, {0}, true, false};
    graph.tensor_descr_map[y] = {{2, 3}, NO_PRODUCER, {0}, true, false};
    graph.tensor_descr_map[out] = {{2, 3}, 0, {}, false, false};

    AddNode addNode;
    addNode.name = "add";
    addNode.input_tensors = {x, y};
    addNode.output_tensors = {out};

    graph.nodes.push_back(addNode);
    graph.tensor_descr_map[out].producer_node_id = 0;
    graph.tensor_descr_map[x].consumer_node_ids = {0};
    graph.tensor_descr_map[y].consumer_node_ids = {0};

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}

// ================================================================
// Тест 5: Простой граф с Mul
// ================================================================
TEST_F(MLIRGeneratorTest, SimpleMulGraph) {
    ComputeGraph graph;

    TensorID x = "x", y = "y", out = "out";

    graph.tensor_descr_map[x] = {{2, 3}, NO_PRODUCER, {0}, true, false};
    graph.tensor_descr_map[y] = {{2, 3}, NO_PRODUCER, {0}, true, false};
    graph.tensor_descr_map[out] = {{2, 3}, 0, {}, false, false};

    MulNode mulNode;
    mulNode.name = "mul";
    mulNode.input_tensors = {x, y};
    mulNode.output_tensors = {out};

    graph.nodes.push_back(mulNode);
    graph.tensor_descr_map[out].producer_node_id = 0;
    graph.tensor_descr_map[x].consumer_node_ids = {0};
    graph.tensor_descr_map[y].consumer_node_ids = {0};

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}
