// tests/test_mlir_passes.cpp
#include <gtest/gtest.h>
#include "graph.hpp"
#include "MiddleEnd/MLIR/MLIRGenerator.hpp"
#include "MiddleEnd/Pipeline/MLIRPasses.hpp"

using namespace tcc;
using namespace tcc::mlir_gen;

TEST(MLIRPassesTest, LoweringPipeline) {
    ComputeGraph graph;

    TensorID a = "a", b = "b", out = "out";

    graph.tensor_descr_map[a] = {{2, 2}, NO_PRODUCER, {0}, true, false};
    graph.tensor_descr_map[b] = {{2, 2}, NO_PRODUCER, {0}, true, false};
    graph.tensor_descr_map[out] = {{2, 2}, 0, {}, false, false};

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
