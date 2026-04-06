// tests/test_matmul_emitter.cpp
#include <gtest/gtest.h>
#include "graph.hpp"
#include "MiddleEnd/MLIR/MLIRGenerator.hpp"

using namespace tcc;
using namespace tcc::mlir_gen;

class MatMulEmitterTest : public ::testing::Test {
protected:
    // 2D матричное умножение: (M x K) * (K x N) -> (M x N)
    ComputeGraph createMatMulGraph(size_t M, size_t K, size_t N) {
        ComputeGraph graph;

        TensorID A = "A", B = "B", C = "C";

        graph.tensor_map[A] = {{M, K}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[B] = {{K, N}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[C] = {{M, N}, 0, {}, false, false};

        MatmulNode matmulNode;
        matmulNode.name = "matmul1";
        matmulNode.input_tensors = {A, B};
        matmulNode.output_tensors = {C};

        graph.nodes.push_back(matmulNode);
        graph.tensor_map[C].producer_node_id = 0;
        graph.tensor_map[A].consumer_node_ids = {0};
        graph.tensor_map[B].consumer_node_ids = {0};

        return graph;
    }

    // Batch матричное умножение: (Batch x M x K) * (Batch x K x N) -> (Batch x M x N)
    ComputeGraph createBatchMatMulGraph(size_t Batch, size_t M, size_t K, size_t N) {
        ComputeGraph graph;

        TensorID A = "A", B = "B", C = "C";

        graph.tensor_map[A] = {{Batch, M, K}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[B] = {{Batch, K, N}, NO_PRODUCER, {}, true, false};
        graph.tensor_map[C] = {{Batch, M, N}, 0, {}, false, false};

        MatmulNode matmulNode;
        matmulNode.name = "matmul1";
        matmulNode.input_tensors = {A, B};
        matmulNode.output_tensors = {C};

        graph.nodes.push_back(matmulNode);
        graph.tensor_map[C].producer_node_id = 0;
        graph.tensor_map[A].consumer_node_ids = {0};
        graph.tensor_map[B].consumer_node_ids = {0};

        return graph;
    }
};

TEST_F(MatMulEmitterTest, MatMul2D) {
    auto graph = createMatMulGraph(2, 3, 4);

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}

TEST_F(MatMulEmitterTest, MatMulSquare) {
    auto graph = createMatMulGraph(4, 4, 4);

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}

TEST_F(MatMulEmitterTest, BatchMatMul3D) {
    auto graph = createBatchMatMulGraph(5, 2, 3, 4);

    MLIRGenerator::Config cfg;
    cfg.printMLIR = true;

    MLIRGenerator generator(cfg);
    EXPECT_TRUE(generator.generate(graph));
}
