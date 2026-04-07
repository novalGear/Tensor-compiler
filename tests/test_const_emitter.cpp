// tests/test_constant_emitter_standalone.cpp
#include <gtest/gtest.h>
#include "graph.hpp"
#include "MiddleEnd/MLIR/MLIRGenerator.hpp"

using namespace tcc;
using namespace tcc::mlir_gen;

TEST(ConstantEmitterStandalone, CreateScalar) {
    ComputeGraph graph;

    TensorID out = "out";

    graph.tensor_descr_map[out] = {{}, 0, {}, false, false};

    ConstantNode constNode;
    constNode.name = "const";
    constNode.output_tensors = {out};
    constNode.value = {42.0f};

    graph.nodes.push_back(constNode);
    graph.tensor_descr_map[out].producer_node_id = 0;

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    ASSERT_TRUE(generator.generate(graph));
}

TEST(ConstantEmitterStandalone, CreateTensor2x2) {
    ComputeGraph graph;

    TensorID out = "out";

    graph.tensor_descr_map[out] = {{2, 2}, 0, {}, false, false};

    ConstantNode constNode;
    constNode.name = "const";
    constNode.output_tensors = {out};
    constNode.value = {1.0f, 2.0f, 3.0f, 4.0f};

    graph.nodes.push_back(constNode);
    graph.tensor_descr_map[out].producer_node_id = 0;

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    ASSERT_TRUE(generator.generate(graph));
}

TEST(ConstantEmitterStandalone, ConstAndAdd) {
    ComputeGraph graph;

    TensorID input = "input";
    TensorID const_out = "const_out";
    TensorID output = "output";

    // Вход
    graph.tensor_descr_map[input] = {{2, 2}, NO_PRODUCER, {1}, true, false};

    // Константа
    graph.tensor_descr_map[const_out] = {{2, 2}, 0, {1}, false, false};

    // Выход
    graph.tensor_descr_map[output] = {{2, 2}, 1, {}, false, false};

    // Узел 0: Constant
    ConstantNode constNode;
    constNode.name = "const";
    constNode.output_tensors = {const_out};
    constNode.value = {1.0f, 2.0f, 3.0f, 4.0f};

    // Узел 1: Add
    AddNode addNode;
    addNode.name = "add";
    addNode.input_tensors = {input, const_out};
    addNode.output_tensors = {output};

    graph.nodes.push_back(constNode);
    graph.nodes.push_back(addNode);

    graph.tensor_descr_map[const_out].producer_node_id = 0;
    graph.tensor_descr_map[output].producer_node_id = 1;
    graph.tensor_descr_map[input].consumer_node_ids = {1};
    graph.tensor_descr_map[const_out].consumer_node_ids = {1};

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    ASSERT_TRUE(generator.generate(graph));
}
