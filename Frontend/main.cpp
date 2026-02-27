#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <variant>
#include <set>

#include "graph.hpp"

// Функция сохранения графа в DOT формат
void save_dot(const tcc::ComputeGraph& graph, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[Error] Cannot create file: " << filename << std::endl;
        return;
    }

    out << "digraph Model {\n";
    out << "  rankdir=TB;\n";
    out << "  node [shape=box, style=filled, fillcolor=lightblue];\n";
    out << "  edge [fontsize=10];\n\n";

    // 1. Рисуем узлы
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];

        // Получаем тип операции
        std::string op_type = std::visit([](const auto& n) -> std::string {
            using T = std::decay_t<decltype(n)>;
            if constexpr (std::is_same_v<T, tcc::AddNode>) return "Add";
            if constexpr (std::is_same_v<T, tcc::MulNode>) return "Mul";
            if constexpr (std::is_same_v<T, tcc::ReluNode>) return "Relu";
            if constexpr (std::is_same_v<T, tcc::MatmulNode>) return "MatMul";
            if constexpr (std::is_same_v<T, tcc::GemmNode>) return "Gemm";
            if constexpr (std::is_same_v<T, tcc::ConvNode>) return "Conv";
            return "Unknown";
        }, node);

        // Получаем имя узла
        std::string name = std::visit([](const auto& n) { return n.name; }, node);
        if (name.empty()) name = "node_" + std::to_string(i);

        // Формируем подпись (Тип \n Имя)
        std::string label = op_type + "\\n" + name;

        out << "  n" << i << " [label=\"" << label << "\"];\n";
    }

    out << "\n";

    // 2. Рисуем связи (ребра)
    // Используем set, чтобы не дублировать объявления входных тензоров
    std::set<std::string> drawn_inputs;

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];

        std::visit([&](const auto& n) {
            for (const auto& input_name : n.input_tensors) {
                auto it = graph.tensor_map.find(input_name);

                // Если тензор найден в карте
                if (it != graph.tensor_map.end()) {
                    // Проверяем, есть ли у него производитель внутри графа
                    if (it->second.producer_node_id != tcc::NO_PRODUCER) {
                        size_t src_id = it->second.producer_node_id;
                        out << "  n" << src_id << " -> n" << i << " [label=\"" << input_name << "\"];\n";
                    } else {
                        // Это внешний вход (Input Graph)
                        // Рисуем узел входа, если еще не рисовали
                        if (drawn_inputs.find(input_name) == drawn_inputs.end()) {
                            out << "  \"" << input_name << "\" [shape=ellipse, style=dashed, fillcolor=white];\n";
                            drawn_inputs.insert(input_name);
                        }
                        out << "  \"" << input_name << "\" -> n" << i << " [label=\"" << input_name << "\"];\n";
                    }
                }
            }
        }, node);
    }

    out << "}\n";
    out.close();
    std::cout << "[Info] DOT file saved to: " << filename << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== Minimal ONNX Loader ===" << std::endl;

    std::string model_path = "test_model.onnx";
    if (argc > 1) {
        model_path = argv[1];
    } else {
        if (!std::filesystem::exists(model_path)) model_path = "../data/test_model.onnx";
        if (!std::filesystem::exists(model_path)) model_path = "../../data/test_model.onnx";
    }

    if (!std::filesystem::exists(model_path)) {
        std::cerr << "[ERROR] Model not found: " << model_path << std::endl;
        return 1;
    }

    std::cout << "Loading: " << model_path << "..." << std::endl;

    auto graph = tcc::ComputeGraph::load_from_onnx(model_path);

    if (!graph) {
        std::cerr << "[ERROR] Failed to load model!" << std::endl;
        return 1;
    }

    std::cout << "[SUCCESS] Model loaded." << std::endl;
    std::cout << "Nodes: " << graph->nodes.size() << ", Tensors: " << graph->tensor_map.size() << std::endl;

    // Сохраняем в DOT
    save_dot(*graph, "output.dot");

    std::cout << "\nTo visualize, run:" << std::endl;
    std::cout << "  dot -Tpng output.dot -o graph.png && xdg-open graph.png" << std::endl;

    return 0;
}
