#include "graph.hpp"
// Подключаем сгенерированные утилиты (цвет + лейбл)
#include "graph_gen_utils.inl"

#include <fstream>
#include <iostream>
#include <set>
#include <string>

namespace tcc {

void save_dot(const ComputeGraph& graph, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[Error] Cannot create file: " << filename << std::endl;
        return;
    }

    out << "digraph Model {\n";
    out << "  rankdir=TB;\n"; // Top to Bottom
    out << "  bgcolor=\"white\";\n";
    // Настройки узлов по умолчанию
    out << "  node [shape=record, style=\"rounded,filled\", fontname=\"Arial\", fontsize=10, margin=0.1];\n";
    out << "  edge [fontname=\"Arial\", fontsize=9, arrowsize=0.7];\n\n";

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];

        // 1. Получаем тип операции (для цвета)
        std::string op_type = std::visit([](const auto& n) -> std::string {
            using T = std::decay_t<decltype(n)>;
            if constexpr (std::is_same_v<T, AddNode>) return "Add";
            if constexpr (std::is_same_v<T, MulNode>) return "Mul";
            if constexpr (std::is_same_v<T, ReluNode>) return "Relu";
            if constexpr (std::is_same_v<T, MatmulNode>) return "MatMul";
            if constexpr (std::is_same_v<T, GemmNode>) return "Gemm";
            if constexpr (std::is_same_v<T, ConvNode>) return "Conv";
            if constexpr (std::is_same_v<T, ConstantNode>) return "Constant";
            else return "Unknown";
        }, node);

        // 2. Получаем цвет из сгенерированной функции
        std::string color = get_node_color(op_type);

        // 3. Получаем HTML-лейбл из сгенерированной функции
        std::string label_content = get_node_record_label(node);

        // Формируем итоговую строку атрибута label=<...>
        // Важно: в DOT HTML-лейблы заключаются в угловые скобки < >
        out << "  n" << i << " [label=\"" << label_content << "\", fillcolor=\"" << color << "\"];\n";
    }

    out << "\n";

    // Отрисовка связей
    std::set<std::string> drawn_inputs;
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];

        std::visit([&](const auto& n) {
            for (const auto& in : n.input_tensors) {
                auto it = graph.tensor_descr_map.find(in);
                if (it != graph.tensor_descr_map.end()) {
                    if (it->second.producer_node_id != NO_PRODUCER) {
                        size_t src_id = it->second.producer_node_id;
                        out << "  n" << src_id << " -> n" << i << " [label=\"" << in << "\"];\n";
                    } else {
                        // Входной тензор
                        if (drawn_inputs.find(in) == drawn_inputs.end()) {
                            out << "  \"" << in << "\" [shape=ellipse, style=dashed, fillcolor=\"white\", label=\"" << in << "\"];\n";
                            drawn_inputs.insert(in);
                        }
                        out << "  \"" << in << "\" -> n" << i << " [label=\"" << in << "\"];\n";
                    }
                }
            }
        }, node);
    }

    out << "}\n";
    out.close();
    std::cout << "[Info] DOT graph saved to: " << filename << std::endl;
}


std::string ComputeGraph::getNodeTypeName(const ComputeNode& node) {
    return std::visit([](const auto& n) -> std::string {
        using T = std::decay_t<decltype(n)>;

        if constexpr (std::is_same_v<T, ConstantNode>) return "Constant";
        else if constexpr (std::is_same_v<T, AddNode>) return "Add";
        else if constexpr (std::is_same_v<T, MulNode>) return "Mul";
        else if constexpr (std::is_same_v<T, ReluNode>) return "Relu";
        else if constexpr (std::is_same_v<T, MatmulNode>) return "MatMul";
        else if constexpr (std::is_same_v<T, GemmNode>) return "Gemm";
        else if constexpr (std::is_same_v<T, ConvNode>) return "Conv";
        else if constexpr (std::is_same_v<T, FlattenNode>) return "Flatten";
        else return "Unknown";
    }, node);
}

void ComputeGraph::print_tensor_descr_map(std::ostream& os) const {
    os << "\n=== Tensor Map (JSON) ===\n";
    os << "{\n  \"tensors\": [\n";

    bool first = true;
    for (const auto& [name, desc] : tensor_descr_map) {
        if (!first) os << ",\n";
        first = false;

        // Форматируем размерности
        std::string dims_str = "[";
        for (size_t d : desc.dimensions) {
            dims_str += std::to_string(d) + ", ";
        }
        if (!desc.dimensions.empty()) {
            dims_str.pop_back(); // убираем пробел
            dims_str.pop_back(); // убираем запятую
        }
        dims_str += "]";

        // Producer тип
        std::string producer_type = "null";
        if (desc.producer_node_id != NO_PRODUCER && desc.producer_node_id < nodes.size()) {
            producer_type = "\"" + getNodeTypeName(nodes[desc.producer_node_id]) + "\"";
        }

        os << "    {\n"
           << "      \"name\": \"" << name << "\",\n"
           << "      \"dims\": " << dims_str << ",\n"
           << "      \"producer\": " << (desc.producer_node_id == NO_PRODUCER ? "null" : std::to_string(desc.producer_node_id)) << ",\n"
           << "      \"producer_type\": " << producer_type << ",\n"
           << "      \"consumers_count\": " << desc.consumer_node_ids.size() << ",\n"
           << "      \"is_graph_input\": " << (desc.is_graph_input ? "true" : "false") << ",\n"
           << "      \"is_initializer\": " << (desc.is_initializer ? "true" : "false") << "\n"
           << "    }";
    }

    os << "\n  ]\n}\n";
}

void ComputeGraph::print_nodes(std::ostream& os) const {
    os << "\n=== Nodes ===\n";
    if (nodes.empty()) {
        os << "  (empty)\n";
        return;
    }

    for (size_t i = 0; i < nodes.size(); ++i) {
        os << "  Node " << i << ": ";

        std::visit([&os, i](const auto& node) {
            // Имя узла
            os << "name='" << node.name << "'";

            // Тип узла
            os << ", type=";
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, ConstantNode>) os << "Constant";
            else if constexpr (std::is_same_v<T, AddNode>) os << "Add";
            else if constexpr (std::is_same_v<T, MulNode>) os << "Mul";
            else if constexpr (std::is_same_v<T, ReluNode>) os << "Relu";
            else if constexpr (std::is_same_v<T, MatmulNode>) os << "MatMul";
            else if constexpr (std::is_same_v<T, GemmNode>) os << "Gemm";
            else if constexpr (std::is_same_v<T, ConvNode>) os << "Conv";
            else if constexpr (std::is_same_v<T, FlattenNode>) os << "Flatten";
            else os << "Unknown";

            // Входы
            os << ", inputs: [";
            for (size_t j = 0; j < node.input_tensors.size(); ++j) {
                if (j > 0) os << ", ";
                os << node.input_tensors[j];
            }
            os << "]";

            // Выходы
            os << ", outputs: [";
            for (size_t j = 0; j < node.output_tensors.size(); ++j) {
                if (j > 0) os << ", ";
                os << node.output_tensors[j];
            }
            os << "]";

            // Специфичные для типа узла данные
            if constexpr (std::is_same_v<T, ConstantNode>) {
                os << ", value_size: " << node.value.size();
            }
            else if constexpr (std::is_same_v<T, GemmNode>) {
                os << ", transA=" << node.transposeA << ", transB=" << node.transposeB;
                os << ", alpha=" << node.alpha << ", beta=" << node.beta;
            }
            else if constexpr (std::is_same_v<T, ConvNode>) {
                os << ", strides=[";
                for (size_t j = 0; j < node.strides.size(); ++j) {
                    if (j > 0) os << ",";
                    os << node.strides[j];
                }
                os << "], group=" << node.group;
            }
            else if constexpr (std::is_same_v<T, FlattenNode>) {
                os << ", axis=" << node.axis;
            }
        }, nodes[i]);

        os << "\n";
    }
}

void ComputeGraph::print_inputs_outputs(std::ostream& os) const {
    auto inputs = collectInputs();
    auto outputs = collectOutputs();

    os << "\n=== Graph Inputs (" << inputs.size() << ") ===\n";
    if (inputs.empty()) {
        os << "  (none)\n";
    } else {
        for (const auto& in : inputs) {
            auto it = tensor_descr_map.find(in);
            os << "  " << in;
            if (it != tensor_descr_map.end() && !it->second.dimensions.empty()) {
                os << " dims: [";
                for (size_t d : it->second.dimensions) {
                    os << d << " ";
                }
                os << "]";
            }
            os << "\n";
        }
    }

    os << "\n=== Graph Outputs (" << outputs.size() << ") ===\n";
    if (outputs.empty()) {
        os << "  (none)\n";
    } else {
        for (const auto& out : outputs) {
            auto it = tensor_descr_map.find(out);
            os << "  " << out;
            if (it != tensor_descr_map.end() && !it->second.dimensions.empty()) {
                os << " dims: [";
                for (size_t d : it->second.dimensions) {
                    os << d << " ";
                }
                os << "]";
            }
            os << "\n";
        }
    }
}

void ComputeGraph::print_graph_info(std::ostream& os) const {
    os << "\n" << std::string(60, '=') << "\n";
    os << "COMPUTE GRAPH DIAGNOSTICS\n";
    os << std::string(60, '=') << "\n";

    os << "\n📊 Summary:\n";
    os << "  Nodes: " << nodes.size() << "\n";
    os << "  Tensors: " << tensor_descr_map.size() << "\n";

    print_inputs_outputs(os);
    print_tensor_descr_map(os);
    print_nodes(os);

    os << "\n" << std::string(60, '=') << "\n";
}

} // namespace tcc
