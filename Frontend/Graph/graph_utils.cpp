#include "graph.hpp"
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <variant>

namespace tcc {

// Хелпер для получения читаемого имени типа операции
std::string get_op_type_name(const ComputeNode& node) {
    return std::visit([](const auto& n) -> std::string {
        using T = std::decay_t<decltype(n)>;
        if constexpr (std::is_same_v<T, AddNode>) return "Add";
        if constexpr (std::is_same_v<T, MulNode>) return "Mul";
        if constexpr (std::is_same_v<T, ReluNode>) return "Relu";
        if constexpr (std::is_same_v<T, MatmulNode>) return "MatMul";
        if constexpr (std::is_same_v<T, GemmNode>) return "Gemm";
        if constexpr (std::is_same_v<T, ConvNode>) return "Conv";
        else return "Unknown";
    }, node);
}

void save_dot(const ComputeGraph& graph, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[Error] Could not create file: " << filename << std::endl;
        return;
    }

    out << "digraph Model {\n";
    out << "  rankdir=TB;\n"; // Top to Bottom
    out << "  node [shape=box, style=filled, fillcolor=lightblue, fontname=\"Arial\"];\n";
    out << "  edge [fontname=\"Arial\", fontsize=10];\n\n";

    // 1. Рисуем узлы
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        std::string op_type = get_op_type_name(node);

        // Доступ к полю name возможен напрямую, так как оно есть у всех структур
        // Но нам нужно получить его через visit, так как node - это variant
        std::string node_name = std::visit([](const auto& n) { return n.name; }, node);

        std::string label = op_type;
        if (!node_name.empty()) {
            label += "\\n" + node_name;
        }

        // Опционально: добавим специфичные атрибуты в подпись для Conv и Gemm
        std::visit([&](const auto& n) {
            if constexpr (std::is_same_v<std::decay_t<decltype(n)>, ConvNode>) {
                if (!n.strides.empty()) {
                    label += "\\nstrides=[";
                    for (size_t k = 0; k < n.strides.size(); ++k) {
                        if (k > 0) label += ",";
                        label += std::to_string(n.strides[k]);
                    }
                    label += "]";
                }
            }
            if constexpr (std::is_same_v<std::decay_t<decltype(n)>, GemmNode>) {
                if (n.transposeA || n.transposeB) {
                    label += "\\ntA=" + std::to_string(n.transposeA) +
                             ", tB=" + std::to_string(n.transposeB);
                }
            }
        }, node);

        out << "  n" << i << " [label=\"" << label << "\"];\n";
    }

    out << "\n";

    // 2. Рисуем связи (ребра)
    std::set<std::string> drawn_inputs;

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];

        std::visit([&](const auto& n) {
            for (const auto& input_name : n.input_tensors) {
                auto it = graph.tensor_map.find(input_name);

                if (it != graph.tensor_map.end()) {
                    if (it->second.producer_node_id != NO_PRODUCER) {
                        // Внутренняя связь
                        size_t src_id = it->second.producer_node_id;
                        out << "  n" << src_id << " -> n" << i
                            << " [label=\"" << input_name << "\"];\n";
                    } else {
                        // Внешний вход (Input Graph)
                        if (drawn_inputs.find(input_name) == drawn_inputs.end()) {
                            out << "  \"" << input_name << "\" [shape=ellipse, style=dashed, fillcolor=white, label=\"" << input_name << "\"];\n";
                            drawn_inputs.insert(input_name);
                        }
                        out << "  \"" << input_name << "\" -> n" << i
                            << " [label=\"" << input_name << "\"];\n";
                    }
                }
            }
        }, node);
    }

    out << "}\n";
    out.close();
    std::cout << "[Info] DOT graph saved to: " << filename << std::endl;
}

} // namespace tcc
