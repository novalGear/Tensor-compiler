#include <iostream>
#include <string>
#include <filesystem>
#include "graph.hpp"

// Объявление функции из graph_utils.cpp
namespace tcc {
    void save_dot(const ComputeGraph& graph, const std::string& filename);
}

int main(int argc, char** argv) {
    std::cout << "=== Tensor Compiler Frontend (Generated) ===" << std::endl;

    // 1. Определение пути к модели
    std::string model_path = "test_model.onnx";
    if (argc > 1) {
        model_path = argv[1];
    } else {
        if (!std::filesystem::exists(model_path)) model_path = "../data/test_model.onnx";
        if (!std::filesystem::exists(model_path)) model_path = "../../data/test_model.onnx";
    }

    if (!std::filesystem::exists(model_path)) {
        std::cerr << "[ERROR] Model not found: " << model_path << std::endl;
        std::cerr << "Usage: " << argv[0] << " [path_to_model.onnx]" << std::endl;
        return 1;
    }

    std::cout << "Loading: " << model_path << "..." << std::endl;

    // 2. Загрузка графа
    auto graph = tcc::ComputeGraph::load_from_onnx(model_path);

    if (!graph) {
        std::cerr << "[FATAL] Failed to load model!" << std::endl;
        return 1;
    }

    std::cout << "[SUCCESS] Model loaded." << std::endl;
    std::cout << "  Nodes: " << graph->nodes.size() << std::endl;
    std::cout << "  Tensors: " << graph->tensor_map.size() << std::endl;

    // 3. Дамп в DOT
    tcc::save_dot(*graph, "data/output.dot");

    std::cout << "\nTo visualize, run:" << std::endl;
    std::cout << "  dot -Tpng output.dot -o graph.png && xdg-open graph.png" << std::endl;

    return 0;
}
