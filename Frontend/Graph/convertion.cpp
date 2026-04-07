#include "graph.hpp"
#include "graph_gen_parser.inl"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <optional>

#include "plog/Log.h"

namespace tcc {

std::unique_ptr<ComputeGraph> ComputeGraph::load_from_onnx(const std::string& filepath) {
    onnx::ModelProto model;
    PLOG_INFO << " Loading ONNX model from: " << filepath << std::endl;

    if (!read_from_onnx_proto(filepath, model)) {
        PLOG_ERROR << " Failed to read ONNX file." << std::endl;
        return nullptr;
    }
    return convertion(model);
}

bool ComputeGraph::read_from_onnx_proto(const std::string& filepath, onnx::ModelProto& model_out) {
    std::ifstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
        PLOG_ERROR << " Failed to open file: " << filepath;
        return false;
    }
    // ParseFromIstream возвращает true при успехе, false при ошибке формата
    if (!model_out.ParseFromIstream(&file)) {
        PLOG_ERROR  << " Failed to parse ONNX model. File might be corrupted or invalid.";
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
    graph->inferAllTensorShapes();
    graph->create_constant_nodes_from_initializers(gp);
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

    PLOG_WARNING << "[Warning] No data found in initializer: " << initializer.name()
              << ", filling with zeros";

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
            PLOG_INFO << " Tensor " << tensor_name
                      << " already has producer, skipping constant creation";
            continue;
        }

        std::vector<float> data = extractInitializerData(init);
        ConstantNode const_node = makeConstantNode(tensor_name, std::move(data));
        NodeID node_id = addNode(std::move(const_node));
        updateTensorMapForConstant(tensor_name, node_id, init);

        PLOG_INFO << "Created ConstantNode for: " << tensor_name
                  << " (size: " << data.size() << ", node_id: " << node_id << ")";
    }
}

// graph.cpp - реализация

//==============================================================================
// SETTERS & HELPERS
//==============================================================================

void ComputeGraph::setTensorDims(const TensorID& tensor, const std::vector<size_t>& dims) {
    auto it = tensor_descr_map.find(tensor);
    if (it != tensor_descr_map.end()) {
        it->second.dimensions = dims;
    } else {
        TensorDescription desc;
        desc.dimensions = dims;
        tensor_descr_map[tensor] = desc;
    }
}

bool ComputeGraph::needsShapeInference(const TensorID& tensor) const {
    auto it = tensor_descr_map.find(tensor);
    if (it == tensor_descr_map.end()) return true;
    return it->second.dimensions.empty();
}

std::vector<size_t> ComputeGraph::broadcastDims(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    size_t rank = std::max(a.size(), b.size());
    std::vector<size_t> result(rank);

    for (size_t i = 0; i < rank; ++i) {
        size_t dimA = (i < a.size()) ? a[a.size() - 1 - i] : 1;
        size_t dimB = (i < b.size()) ? b[b.size() - 1 - i] : 1;

        if (dimA == dimB || dimA == 1 || dimB == 1) {
            result[rank - 1 - i] = std::max(dimA, dimB);
        } else {
            PLOG_ERROR << "Cannot broadcast dimensions " << dimA << " and " << dimB;
            return {};
        }
    }

    return result;
}

//==============================================================================
// SHAPE INFERENCE FOR EACH OPERATION
//==============================================================================

void ComputeGraph::inferMatMulShape(const MatmulNode& node, const TensorID& outputTensor) {
    auto lhsDims = getTensorDims(node.input_tensors[0]);
    auto rhsDims = getTensorDims(node.input_tensors[1]);

    if (lhsDims.size() < 2 || rhsDims.size() < 2) {
        PLOG_ERROR << " MatMul requires at least 2D inputs";
        return;
    }

    std::vector<size_t> outDims;

    // Основные размерности: M x N
    outDims.push_back(lhsDims[lhsDims.size() - 2]);
    outDims.push_back(rhsDims[rhsDims.size() - 1]);

    // Batch размерности
    for (size_t i = 0; i < lhsDims.size() - 2; ++i) {
        if (i < rhsDims.size() - 2) {
            if (lhsDims[i] == rhsDims[i] || lhsDims[i] == 1 || rhsDims[i] == 1) {
                outDims.insert(outDims.begin(), std::max(lhsDims[i], rhsDims[i]));
            } else {
                PLOG_ERROR << " Batch dimension mismatch in MatMul";
                return;
            }
        } else {
            outDims.insert(outDims.begin(), lhsDims[i]);
        }
    }

    setTensorDims(outputTensor, outDims);
}

void ComputeGraph::inferAddShape(const AddNode& node, const TensorID& outputTensor) {
    auto lhsDims = getTensorDims(node.input_tensors[0]);
    auto rhsDims = getTensorDims(node.input_tensors[1]);

    if (!lhsDims.empty() && !rhsDims.empty()) {
        setTensorDims(outputTensor, broadcastDims(lhsDims, rhsDims));
    } else if (!lhsDims.empty()) {
        setTensorDims(outputTensor, lhsDims);
    } else if (!rhsDims.empty()) {
        setTensorDims(outputTensor, rhsDims);
    }
}

void ComputeGraph::inferMulShape(const MulNode& node, const TensorID& outputTensor) {
    // Same as Add - broadcast
    auto lhsDims = getTensorDims(node.input_tensors[0]);
    auto rhsDims = getTensorDims(node.input_tensors[1]);

    if (!lhsDims.empty() && !rhsDims.empty()) {
        setTensorDims(outputTensor, broadcastDims(lhsDims, rhsDims));
    } else if (!lhsDims.empty()) {
        setTensorDims(outputTensor, lhsDims);
    } else if (!rhsDims.empty()) {
        setTensorDims(outputTensor, rhsDims);
    }
}

void ComputeGraph::inferReluShape(const ReluNode& node, const TensorID& outputTensor) {
    auto inputDims = getTensorDims(node.input_tensors[0]);
    if (!inputDims.empty()) {
        setTensorDims(outputTensor, inputDims);
    }
}

void ComputeGraph::inferGemmShape(const GemmNode& node, const TensorID& outputTensor) {
    auto aDims = getTensorDims(node.input_tensors[0]);
    auto bDims = getTensorDims(node.input_tensors[1]);

    if (aDims.size() < 2 || bDims.size() < 2) {
        PLOG_ERROR << " Gemm requires at least 2D inputs\n";
        return;
    }

    std::vector<size_t> outDims = {aDims[aDims.size() - 2], bDims[bDims.size() - 1]};

    // Batch размерности
    for (size_t i = 0; i < aDims.size() - 2; ++i) {
        if (i < bDims.size() - 2) {
            outDims.insert(outDims.begin(), std::max(aDims[i], bDims[i]));
        } else {
            outDims.insert(outDims.begin(), aDims[i]);
        }
    }

    setTensorDims(outputTensor, outDims);
}

void ComputeGraph::inferConvShape(const ConvNode& node, const TensorID& outputTensor) {
    auto inputDims = getTensorDims(node.input_tensors[0]);
    auto weightDims = getTensorDims(node.input_tensors[1]);

    if (inputDims.size() != 4 || weightDims.size() != 4) {
        PLOG_ERROR << " Conv requires 4D inputs (NCHW)\n";
        return;
    }

    size_t N = inputDims[0];
    size_t OC = weightDims[0];
    size_t H = inputDims[2];
    size_t W = inputDims[3];
    size_t KH = weightDims[2];
    size_t KW = weightDims[3];

    size_t padH = (node.pads.size() > 0) ? node.pads[0] : 0;
    size_t padW = (node.pads.size() > 1) ? node.pads[1] : 0;
    size_t strideH = (node.strides.size() > 0) ? node.strides[0] : 1;
    size_t strideW = (node.strides.size() > 1) ? node.strides[1] : 1;
    size_t dilationH = (node.dilations.size() > 0) ? node.dilations[0] : 1;
    size_t dilationW = (node.dilations.size() > 1) ? node.dilations[1] : 1;

    size_t OH = (H + 2*padH - dilationH*(KH-1) - 1) / strideH + 1;
    size_t OW = (W + 2*padW - dilationW*(KW-1) - 1) / strideW + 1;

    setTensorDims(outputTensor, {N, OC, OH, OW});
}

void ComputeGraph::inferFlattenShape(const FlattenNode& node, const TensorID& outputTensor) {
    auto inputDims = getTensorDims(node.input_tensors[0]);
    if (inputDims.empty()) return;

    int axis = node.axis;
    if (axis < 0) axis = static_cast<int>(inputDims.size()) + axis;

    size_t outer = 1;
    size_t inner = 1;

    for (size_t i = 0; i < inputDims.size(); ++i) {
        if (i < static_cast<size_t>(axis)) {
            outer *= inputDims[i];
        } else {
            inner *= inputDims[i];
        }
    }

    setTensorDims(outputTensor, {outer, inner});
}

//==============================================================================
// MAIN INFERENCE LOOP
//==============================================================================

void ComputeGraph::inferAllTensorShapes() {
    auto order = topologicalSort(false);

    for (size_t nodeId : order) {
        const auto& node = nodes[nodeId];
        std::visit([this](const auto& n) {
            using T = std::decay_t<decltype(n)>;

            if (n.output_tensors.empty()) return;

            const auto& outputTensor = n.output_tensors[0];
            if (!needsShapeInference(outputTensor)) return;

            if constexpr (std::is_same_v<T, MatmulNode>) {
                inferMatMulShape(n, outputTensor);
            }
            else if constexpr (std::is_same_v<T, AddNode>) {
                inferAddShape(n, outputTensor);
            }
            else if constexpr (std::is_same_v<T, MulNode>) {
                inferMulShape(n, outputTensor);
            }
            else if constexpr (std::is_same_v<T, ReluNode>) {
                inferReluShape(n, outputTensor);
            }
            else if constexpr (std::is_same_v<T, GemmNode>) {
                inferGemmShape(n, outputTensor);
            }
            else if constexpr (std::is_same_v<T, ConvNode>) {
                inferConvShape(n, outputTensor);
            }
            else if constexpr (std::is_same_v<T, FlattenNode>) {
                inferFlattenShape(n, outputTensor);
            }
            else if constexpr (std::is_same_v<T, ConstantNode>) {
                // Константа уже имеет размерности из initializer
            }

        }, node);
    }
}

} // namespace tcc
