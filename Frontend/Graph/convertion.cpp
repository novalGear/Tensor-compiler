#include "graph.hpp"
#include "graph_gen_parser.inl"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <optional>

namespace tcc {

std::unique_ptr<ComputeGraph> ComputeGraph::load_from_onnx(const std::string& filepath) {
    onnx::ModelProto model;
    std::cout << "[Info] Loading ONNX model from: " << filepath << std::endl;

    if (!read_from_onnx_proto(filepath, model)) {
        std::cerr << "[Error] Failed to read ONNX file." << std::endl;
        return nullptr;
    }
    return convertion(model);
}

bool ComputeGraph::read_from_onnx_proto(const std::string& filepath, onnx::ModelProto& model_out) {
    std::ifstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "[Error] Failed to open file: " << filepath << std::endl;
        return false;
    }
    // ParseFromIstream возвращает true при успехе, false при ошибке формата
    if (!model_out.ParseFromIstream(&file)) {
        std::cerr   << "[Error] Failed to parse ONNX model. File might be corrupted or invalid."
                    << std::endl;
        return false;
    }
    return true;
}


std::unique_ptr<ComputeGraph> ComputeGraph::convertion(const onnx::ModelProto& model_proto) {
    auto graph = std::make_unique<ComputeGraph>();
    const auto& gp = model_proto.graph();

    graph->register_graph_inputs(gp);
    graph->register_initializers(gp);

    graph->build_nodes(gp);
    graph->fill_tensor_shapes(gp);
    return graph;
}

// ================================================================================================//
// Методы-помощники экземпляра
// ================================================================================================//


void ComputeGraph::register_graph_inputs(const onnx::GraphProto& gp) {
    for (const auto& input : gp.input()) {
        TensorDescription desc;
        desc.producer_node_id = NO_PRODUCER;
        desc.is_graph_input = true;

        if (input.type().has_tensor_type()) {
            const auto& shape = input.type().tensor_type().shape();
            for (int i = 0; i < shape.dim_size(); ++i) {
                desc.dimensions.push_back(shape.dim(i).has_dim_value()
                    ? static_cast<size_t>(shape.dim(i).dim_value()) : 0);
            }
        }
        tensor_map[input.name()] = desc;
    }
}

void ComputeGraph::register_initializers(const onnx::GraphProto& gp) {
    for (const auto& init : gp.initializer()) {
        TensorDescription desc;
        // desc.is_initializer = true;
        for (auto dim : init.dims()) {
            desc.dimensions.push_back(static_cast<size_t>(dim));
        }
        auto it = tensor_map.find(init.name());
        if (it != tensor_map.end()) {
            // it->second.is_initializer = true;
            if (!desc.dimensions.empty()) {
                it->second.dimensions = desc.dimensions;
            }
        } else {
            tensor_map[init.name()] = desc;
        }
    }
}

void ComputeGraph::build_nodes(const onnx::GraphProto& gp) {
    for (int i = 0; i < gp.node_size(); ++i) {
        const auto& node_proto = gp.node(i);
        const std::string& op_type = node_proto.op_type();

        ComputeNode node = create_node_from_proto(node_proto);
        NodeID current_id = nodes.size();
        nodes.push_back(std::move(node));

        update_tensor_connections(current_id, nodes.back());
    }
}

void ComputeGraph::update_tensor_connections(NodeID node_id, const ComputeNode& node) {
    // Лямбда для обхода variant
    auto process = [&](const auto& n) {
        // 1. Узел является производителем (producer) для своих выходов
        for (const auto& out : n.output_tensors) {
            // Если тензора еще нет в карте (редкий случай, но возможный), создаем запись
            if (tensor_map.find(out) == tensor_map.end()) {
                tensor_map[out] = TensorDescription();
            }
            tensor_map[out].producer_node_id = node_id;
        }

        // 2. Узел является потребителем (consumer) для своих входов
        for (const auto& in : n.input_tensors) {
            if (tensor_map.find(in) == tensor_map.end()) {
                // Неявный вход (например, константа, не попавшая в initializers, или ошибка модели)
                tensor_map[in] = TensorDescription();
                tensor_map[in].producer_node_id = NO_PRODUCER;
            }
            tensor_map[in].consumer_node_ids.push_back(node_id);
        }
    };

    std::visit(process, node);
}

void ComputeGraph::fill_tensor_shapes(const onnx::GraphProto& gp) {
    for (const auto& vi : gp.value_info()) {
        auto it = tensor_map.find(vi.name());

        // Заполняем размеры только если они еще неизвестны
        if (it != tensor_map.end() && it->second.dimensions.empty()) {
             if (vi.type().has_tensor_type()) {
                 const auto& shape = vi.type().tensor_type().shape();
                 for (int i = 0; i < shape.dim_size(); ++i) {
                     if (shape.dim(i).has_dim_value()) {
                         it->second.dimensions.push_back(static_cast<size_t>(shape.dim(i).dim_value()));
                     }
                 }
             }
        }
    }
}

} // namespace tcc
