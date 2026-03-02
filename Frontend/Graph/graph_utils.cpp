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
                auto it = graph.tensor_map.find(in);
                if (it != graph.tensor_map.end()) {
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

} // namespace tcc
