#include "graph.hpp"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <optional>

namespace tcc {

NodeID ComputeGraph::addNode(ComputeNode&& node) {
    NodeID node_id = nodes.size();
    nodes.push_back(std::move(node));
    return node_id;
}

ConstantNode ComputeGraph::makeConstantNode(const std::string& tensor_name,
                                             std::vector<float> data) {
    ConstantNode node;
    node.name = tensor_name + "_const";
    node.output_tensors = {tensor_name};
    node.value = std::move(data);
    return node;
}

// ================================================================
// Сбор входных тензоров
// ================================================================
std::vector<TensorID> ComputeGraph::collectInputs() const {
    std::vector<TensorID> inputs;
    for (const auto& [tensorId, tensorDesc] : tensor_descr_map) {
        if (tensorDesc.is_graph_input && !tensorDesc.is_initializer) {
            inputs.push_back(tensorId);
        }
    }
    return inputs;
}

// ================================================================
// Сбор выходных тензоров
// ================================================================
std::vector<TensorID> ComputeGraph::collectOutputs() const {
    std::vector<TensorID> outputs;
    for (const auto& [tensorId, tensorDesc] : tensor_descr_map) {
        if (tensorDesc.consumer_node_ids.empty() &&
            !tensorDesc.is_graph_input &&
            tensorDesc.producer_node_id != NO_PRODUCER) {
            outputs.push_back(tensorId);
        }
    }
    return outputs;
}

// ================================================================
// Получение размерностей тензора
// ================================================================
std::vector<size_t> ComputeGraph::getTensorDims(const TensorID& tensorId) const {
    auto it = tensor_descr_map.find(tensorId);
    if (it != tensor_descr_map.end()) {
        return it->second.dimensions;
    }
    return {};
}

}
