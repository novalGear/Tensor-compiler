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
    graph->create_constant_nodes_from_initializers(gp);  // <-- НОВЫЙ ВЫЗОВ

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
        tensor_descr_map[input.name()] = desc;
    }
}

void ComputeGraph::register_initializers(const onnx::GraphProto& gp) {
    for (const auto& init : gp.initializer()) {
        TensorDescription desc;
        // desc.is_initializer = true;
        for (auto dim : init.dims()) {
            desc.dimensions.push_back(static_cast<size_t>(dim));
        }
        auto it = tensor_descr_map.find(init.name());
        if (it != tensor_descr_map.end()) {
            // it->second.is_initializer = true;
            if (!desc.dimensions.empty()) {
                it->second.dimensions = desc.dimensions;
            }
        } else {
            tensor_descr_map[init.name()] = desc;
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
            if (tensor_descr_map.find(out) == tensor_descr_map.end()) {
                tensor_descr_map[out] = TensorDescription();
            }
            tensor_descr_map[out].producer_node_id = node_id;
        }

        // 2. Узел является потребителем (consumer) для своих входов
        for (const auto& in : n.input_tensors) {
            if (tensor_descr_map.find(in) == tensor_descr_map.end()) {
                // Неявный вход (например, константа, не попавшая в initializers, или ошибка модели)
                tensor_descr_map[in] = TensorDescription();
                tensor_descr_map[in].producer_node_id = NO_PRODUCER;
            }
            tensor_descr_map[in].consumer_node_ids.push_back(node_id);
        }
    };

    std::visit(process, node);
}

void ComputeGraph::fill_tensor_shapes(const onnx::GraphProto& gp) {
    for (const auto& vi : gp.value_info()) {
        auto it = tensor_descr_map.find(vi.name());

        // Заполняем размеры только если они еще неизвестны
        if (it != tensor_descr_map.end() && it->second.dimensions.empty()) {
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

//=================================================================================================
// Работа с initializers
//=================================================================================================

std::vector<float> ComputeGraph::extractInitializerData(const onnx::TensorProto& initializer) {
    std::vector<float> result;

    // 1. Попробуем извлечь из raw_data (самый распространённый случай)
    if (!initializer.raw_data().empty()) {
        const float* float_data = reinterpret_cast<const float*>(initializer.raw_data().data());
        size_t num_elements = initializer.raw_data().size() / sizeof(float);
        result.assign(float_data, float_data + num_elements);
        return result;
    }

    // 2. Попробуем извлечь из float_data
    if (initializer.float_data_size() > 0) {
        result.reserve(initializer.float_data_size());
        for (int i = 0; i < initializer.float_data_size(); ++i) {
            result.push_back(initializer.float_data(i));
        }
        return result;
    }

    // 3. Попробуем извлечь из int32_data (конвертируем в float)
    if (initializer.int32_data_size() > 0) {
        result.reserve(initializer.int32_data_size());
        for (int i = 0; i < initializer.int32_data_size(); ++i) {
            result.push_back(static_cast<float>(initializer.int32_data(i)));
        }
        return result;
    }

    // 4. Попробуем извлечь из int64_data
    if (initializer.int64_data_size() > 0) {
        result.reserve(initializer.int64_data_size());
        for (int i = 0; i < initializer.int64_data_size(); ++i) {
            result.push_back(static_cast<float>(initializer.int64_data(i)));
        }
        return result;
    }

    // 5. Если ничего нет, создаём тензор с нулями
    size_t total_size = 1;
    for (auto dim : initializer.dims()) {
        total_size *= dim;
    }
    result.resize(total_size, 0.0f);

    std::cerr << "[Warning] No data found in initializer: " << initializer.name()
              << ", filling with zeros" << std::endl;

    return result;
}

void ComputeGraph::updateTensorMapForConstant(const std::string& tensor_name,
                                               NodeID node_id,
                                               const onnx::TensorProto& initializer) {
    auto it = tensor_descr_map.find(tensor_name);

    if (it != tensor_descr_map.end()) {
        it->second.producer_node_id = node_id;
        it->second.is_initializer = true;
    } else {
        // Создаём новую запись, если тензор ещё не зарегистрирован
        TensorDescription desc;
        desc.producer_node_id = node_id;
        desc.is_initializer = true;

        // Определяем размерности из initializer
        for (auto dim : initializer.dims()) {
            desc.dimensions.push_back(static_cast<size_t>(dim));
        }

        tensor_descr_map[tensor_name] = desc;
    }
}

void ComputeGraph::create_constant_nodes_from_initializers(const onnx::GraphProto& gp) {
    for (const auto& init : gp.initializer()) {
        const std::string& tensor_name = init.name();
        // Проверяем, есть ли уже producer у этого тензора
        auto it = tensor_descr_map.find(tensor_name);
        if (it != tensor_descr_map.end() && it->second.producer_node_id != NO_PRODUCER) {
            // Уже есть producer (например, это выход какой-то операции)
            std::cout << "[Info] Tensor " << tensor_name
                      << " already has producer, skipping constant creation" << std::endl;
            continue;
        }

        std::vector<float> data = extractInitializerData(init);
        ConstantNode const_node = makeConstantNode(tensor_name, std::move(data));
        NodeID node_id = addNode(std::move(const_node));
        updateTensorMapForConstant(tensor_name, node_id, init);

        std::cout << "[Info] Created ConstantNode for: " << tensor_name
                  << " (size: " << data.size() << ", node_id: " << node_id << ")" << std::endl;
    }
}

} // namespace tcc
