#include "graph.hpp"

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


// TODO: чисто нейросеть писала
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

// TODO: чисто нейросеть писала
void ComputeGraph::register_initializers(const onnx::GraphProto& gp) {
    for (const auto& init : gp.initializer()) {
        TensorDescription desc;
        // desc.is_initializer = true;
        for (auto dim : init.dims()) desc.dimensions.push_back(static_cast<size_t>(dim));

        auto it = tensor_map.find(init.name());
        if (it != tensor_map.end()) {
            // it->second.is_initializer = true;
            if (!desc.dimensions.empty()) it->second.dimensions = desc.dimensions;
        } else {
            tensor_map[init.name()] = desc;
        }
    }
}

void ComputeGraph::build_nodes(const onnx::GraphProto& gp) {
    for (int i = 0; i < gp.node_size(); ++i) {
        const auto& node_proto = gp.node(i);
        const std::string& op_type = node_proto.op_type();

        std::optional<ComputeNode> created_node; // Используем optional для временного хранения

        // Простая и явная логика создания
        if (op_type == "Add") {
            AddNode n(node_proto.name());
            n.load_from_proto(node_proto);
            created_node = ComputeNode(std::move(n));
        }
        else if (op_type == "Mul") {
            MulNode n(node_proto.name());
            n.load_from_proto(node_proto);
            created_node = ComputeNode(std::move(n));
        }
        else if (op_type == "Relu") {
            ReluNode n(node_proto.name());
            n.load_from_proto(node_proto);
            created_node = ComputeNode(std::move(n));
        }
        else if (op_type == "MatMul") {
            MatmulNode n(node_proto.name());
            n.load_from_proto(node_proto);
            created_node = ComputeNode(std::move(n));
        }
        else if (op_type == "Gemm") {
            GemmNode n(node_proto.name());
            n.load_from_proto(node_proto);
            created_node = ComputeNode(std::move(n));
        }
        else if (op_type == "Conv") {
            ConvNode n(node_proto.name());
            n.load_from_proto(node_proto);
            created_node = ComputeNode(std::move(n));
        }
        else {
            std::cerr << "[Warning] Unsupported op: " << op_type
                      << " (" << node_proto.name() << "). Skipping." << std::endl;
            continue;
        }

        // Добавляем узел в граф
        NodeID current_id = nodes.size();
        nodes.push_back(std::move(*created_node));

        // Обновляем связи
        update_tensor_connections(current_id, nodes.back());
    }
}

// TODO: это как вообще работает???
void ComputeGraph::update_tensor_connections(NodeID node_id, const ComputeNode& node) {
    auto process = [&](const auto& n) {
        // Producer для выходов
        for (const auto& out : n.output_tensors) {
            tensor_map[out].producer_node_id = node_id;
        }
        // Consumer для входов
        for (const auto& in : n.input_tensors) {
            if (tensor_map.find(in) == tensor_map.end()) {
                tensor_map[in] = TensorDescription(); // Неявный вход
            }
            tensor_map[in].consumer_node_ids.push_back(node_id);
        }
    };
    std::visit(process, node);
}

// TODO: полностью нейронка писала
void ComputeGraph::fill_tensor_shapes(const onnx::GraphProto& gp) {
    for (const auto& vi : gp.value_info()) {
        auto it = tensor_map.find(vi.name());
        // Заполняем только если размеры еще не известны и запись существует
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
