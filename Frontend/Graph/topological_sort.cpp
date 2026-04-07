// topological_sort.cpp
#include "graph.hpp"
#include <queue>
#include <vector>
#include <iostream>
#include <iomanip>

namespace tcc {

// ================================================================
// Вспомогательная функция для получения имени типа узла
// ================================================================
static std::string getNodeType(const ComputeNode& node) {
    return std::visit([](const auto& n) -> std::string {
        using T = std::decay_t<decltype(n)>;
        if constexpr (std::is_same_v<T, ConstantNode>) return "Const";
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

// ================================================================
// Печать графа зависимостей
// ================================================================
static void printDependencyGraph(const ComputeGraph& graph,
                                  const std::vector<std::vector<size_t>>& deps,
                                  const std::vector<int>& inDegree) {
    std::cout << "\n=== Dependency Graph ===\n";
    std::cout << "Nodes: " << graph.nodes.size() << "\n\n";

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        std::cout << "Node " << std::setw(2) << i << " [" << getNodeType(graph.nodes[i]) << "]";
        std::cout << " (inDegree=" << inDegree[i] << "): ";

        if (deps[i].empty()) {
            std::cout << "no outgoing edges";
        } else {
            std::cout << "-> ";
            for (size_t j = 0; j < deps[i].size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << deps[i][j];
            }
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// ================================================================
// Печать BFS порядка (как он строится)
// ================================================================
static std::vector<size_t> kahnAlgorithmWithDebug(const std::vector<std::vector<size_t>>& deps,
                                                   std::vector<int>& inDegree,
                                                   const ComputeGraph& graph) {
    std::queue<size_t> queue;

    // Начальные узлы (inDegree == 0)
    std::vector<size_t> startNodes;
    for (size_t i = 0; i < inDegree.size(); ++i) {
        if (inDegree[i] == 0) {
            queue.push(i);
            startNodes.push_back(i);
        }
    }

    std::cout << "\n=== Kahn's Algorithm (BFS Topological Sort) ===\n";
    std::cout << "Initial queue (nodes with inDegree=0): ";
    for (size_t node : startNodes) {
        std::cout << node << "[" << getNodeType(graph.nodes[node]) << "] ";
    }
    std::cout << "\n\n";

    std::vector<size_t> order;
    int step = 1;

    while (!queue.empty()) {
        size_t node = queue.front();
        queue.pop();
        order.push_back(node);

        std::cout << "Step " << step++ << ": Pop node " << node
                  << " [" << getNodeType(graph.nodes[node]) << "]\n";
        std::cout << "  Order so far: [";
        for (size_t i = 0; i < order.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << order[i];
        }
        std::cout << "]\n";

        // Обрабатываем зависимости
        if (!deps[node].empty()) {
            std::cout << "  Processing outgoing edges: ";
            for (size_t dep : deps[node]) {
                inDegree[dep]--;
                std::cout << dep << " (inDegree now " << inDegree[dep] << ") ";

                if (inDegree[dep] == 0) {
                    queue.push(dep);
                    std::cout << "→ queued ";
                }
            }
            std::cout << "\n";
        }

        // Показываем текущее состояние очереди
        if (!queue.empty()) {
            std::cout << "  Queue now: ";
            std::queue<size_t> temp = queue;
            while (!temp.empty()) {
                std::cout << temp.front() << "[" << getNodeType(graph.nodes[temp.front()]) << "] ";
                temp.pop();
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    return order;
}

// ================================================================
// Проверка валидности графа
// ================================================================
static bool validateGraph(const ComputeGraph& graph) {
    if (graph.nodes.empty()) {
        std::cerr << "[topo] Graph has no nodes\n";
        return false;
    }

    for (const auto& [tensorId, tensorDesc] : graph.tensor_descr_map) {
        if (tensorDesc.producer_node_id != NO_PRODUCER) {
            if (tensorDesc.producer_node_id >= graph.nodes.size()) {
                std::cerr << "[topo] Producer " << tensorDesc.producer_node_id
                          << " out of range (max " << graph.nodes.size() - 1 << ")\n";
                return false;
            }
        }

        for (NodeID consumer : tensorDesc.consumer_node_ids) {
            if (consumer >= graph.nodes.size()) {
                std::cerr << "[topo] Consumer " << consumer << " out of range (max "
                          << graph.nodes.size() - 1 << ")\n";
                return false;
            }
        }
    }
    return true;
}

// ================================================================
// Построение графа зависимостей
// ================================================================
static void buildDependencies(const ComputeGraph& graph,
                              std::vector<std::vector<size_t>>& deps,
                              std::vector<int>& inDegree) {
    for (const auto& [tensorId, tensorDesc] : graph.tensor_descr_map) {
        if (tensorDesc.producer_node_id == NO_PRODUCER) {
            continue;
        }

        size_t producer = tensorDesc.producer_node_id;
        for (NodeID consumer : tensorDesc.consumer_node_ids) {
            // Избегаем дубликатов
            if (std::find(deps[producer].begin(), deps[producer].end(), consumer) == deps[producer].end()) {
                deps[producer].push_back(consumer);
                inDegree[consumer]++;
            }
        }
    }
}

// ================================================================
// Метод ComputeGraph с отладочной печатью
// ================================================================
std::vector<size_t> ComputeGraph::topologicalSort(bool verbose) const {
    if (nodes.empty()) {
        std::cerr << "[topo] Graph is empty\n";
        return {};
    }

    if (!validateGraph(*this)) {
        return {};
    }

    size_t n = nodes.size();
    std::vector<std::vector<size_t>> deps(n);
    std::vector<int> inDegree(n, 0);

    buildDependencies(*this, deps, inDegree);

    if (verbose) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "TOPOLOGICAL SORT DEBUG\n";
        std::cout << std::string(60, '=') << "\n";

        // Печатаем все узлы с их типами
        std::cout << "\nAll nodes:\n";
        for (size_t i = 0; i < nodes.size(); ++i) {
            std::cout << "  Node " << std::setw(2) << i << ": "
                      << std::setw(10) << getNodeType(nodes[i])
                      << " - " << std::visit([](const auto& n) { return n.name; }, nodes[i]) << "\n";
        }

        printDependencyGraph(*this, deps, inDegree);
    }

    auto order = kahnAlgorithmWithDebug(deps, inDegree, *this);

    if (order.size() != n) {
        std::cerr << "[topo] Cycle detected! Expected " << n << " nodes, got " << order.size() << "\n";

        // Находим непосещённые узлы
        std::vector<bool> visited(n, false);
        for (size_t node : order) visited[node] = true;

        std::cerr << "Unvisited nodes (possible cycle): ";
        for (size_t i = 0; i < n; ++i) {
            if (!visited[i]) {
                std::cerr << i << "[" << getNodeType(nodes[i]) << "] ";
            }
        }
        std::cerr << "\n";

        return {};
    }

    if (verbose) {
        std::cout << "\n=== Final Topological Order ===\n";
        std::cout << "[";
        for (size_t i = 0; i < order.size(); ++i) {
            if (i > 0) std::cout << " → ";
            std::cout << order[i] << "[" << getNodeType(nodes[order[i]]) << "]";
        }
        std::cout << "]\n";
        std::cout << std::string(60, '=') << "\n\n";
    }

    return order;
}

} // namespace tcc
