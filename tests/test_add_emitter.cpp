// tests/test_add_emitter_standalone.cpp
#include <gtest/gtest.h>
#include "graph.hpp"
#include "MiddleEnd/MLIR/MLIRGenerator.hpp"

using namespace tcc;
using namespace tcc::mlir_gen;

TEST(AddEmitterStandalone, Add2x2) {
    ComputeGraph graph;

    TensorID a = "a", b = "b", out = "out";

    // Входы
    graph.tensor_descr_map[a] = {{2, 2}, NO_PRODUCER, {0}, true, false};
    graph.tensor_descr_map[b] = {{2, 2}, NO_PRODUCER, {0}, true, false};
    graph.tensor_descr_map[out] = {{2, 2}, 0, {}, false, false};

    // Add узел
    AddNode addNode;
    addNode.name = "add";
    addNode.input_tensors = {a, b};
    addNode.output_tensors = {out};

    graph.nodes.push_back(addNode);
    graph.tensor_descr_map[out].producer_node_id = 0;
    graph.tensor_descr_map[a].consumer_node_ids = {0};
    graph.tensor_descr_map[b].consumer_node_ids = {0};

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    ASSERT_TRUE(generator.generate(graph));
}
