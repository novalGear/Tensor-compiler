#pragma once

#include "onnx/onnx.pb.h"
// Подключаем сгенерированные структуры и variant
#include "graph_gen.hpp"

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <limits>
#include <iostream>

namespace tcc {

using TensorID = std::string;
using NodeID = size_t;

constexpr NodeID NO_PRODUCER = std::numeric_limits<NodeID>::max();

// Описание тензора
struct TensorDescription {
    std::vector<size_t> dimensions;
    NodeID producer_node_id = NO_PRODUCER;
    std::vector<NodeID> consumer_node_ids;
    bool is_graph_input = false;
    bool is_initializer = false;
};

class ComputeGraph {
public:
    // Хранилище узлов (variant из сгенерированных типов)
    std::vector<ComputeNode> nodes;

    // Карта тензоров
    std::unordered_map<TensorID, TensorDescription> tensor_map;

    static std::unique_ptr<ComputeGraph> load_from_onnx(const std::string& filepath);

private:
    static bool read_from_onnx_proto(const std::string& filepath, onnx::ModelProto& model_out);
    static std::unique_ptr<ComputeGraph> convertion(const onnx::ModelProto& model_proto);

    void register_graph_inputs(const onnx::GraphProto& gp);
    void register_initializers(const onnx::GraphProto& gp);
    void build_nodes(const onnx::GraphProto& gp);
    void update_tensor_connections(NodeID node_id, const ComputeNode& node);
    void fill_tensor_shapes(const onnx::GraphProto& gp);
};

} // namespace tcc
