#include "graph.hpp"
#include <queue>
#include <vector>
#include <iostream>

namespace tcc {

// ================================================================
// Проверка валидности графа
// ================================================================
static bool validateGraph(const ComputeGraph& graph) {
    if (graph.nodes.empty()) {
        return false;
    }

    for (const auto& [tensorId, tensorDesc] : graph.tensor_map) {
        if (tensorDesc.producer_node_id != NO_PRODUCER) {
            if (tensorDesc.producer_node_id >= graph.nodes.size()) {
                std::cerr << "[topo] Producer " << tensorDesc.producer_node_id
                          << " out of range\n";
                return false;
            }
        }

        for (NodeID consumer : tensorDesc.consumer_node_ids) {
            if (consumer >= graph.nodes.size()) {
                std::cerr << "[topo] Consumer " << consumer << " out of range\n";
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
    for (const auto& [tensorId, tensorDesc] : graph.tensor_map) {
        if (tensorDesc.producer_node_id == NO_PRODUCER) {
            continue;
        }

        size_t producer = tensorDesc.producer_node_id;
        for (NodeID consumer : tensorDesc.consumer_node_ids) {
            deps[producer].push_back(consumer);
            inDegree[consumer]++;
        }
    }
}

// ================================================================
// Алгоритм Кана
// ================================================================
static std::vector<size_t> kahnAlgorithm(const std::vector<std::vector<size_t>>& deps,
                                          std::vector<int>& inDegree) {
    std::queue<size_t> queue;
    for (size_t i = 0; i < inDegree.size(); ++i) {
        if (inDegree[i] == 0) {
            queue.push(i);
        }
    }

    std::vector<size_t> order;
    while (!queue.empty()) {
        size_t node = queue.front();
        queue.pop();
        order.push_back(node);

        for (size_t dep : deps[node]) {
            if (--inDegree[dep] == 0) {
                queue.push(dep);
            }
        }
    }
    return order;
}

// ================================================================
// Метод ComputeGraph
// ================================================================
std::vector<size_t> ComputeGraph::topologicalSort() const {
    if (nodes.empty()) {
        return {};
    }

    if (!validateGraph(*this)) {
        return {};
    }

    size_t n = nodes.size();
    std::vector<std::vector<size_t>> deps(n);
    std::vector<int> inDegree(n, 0);

    buildDependencies(*this, deps, inDegree);

    auto order = kahnAlgorithm(deps, inDegree);

    if (order.size() != n) {
        std::cerr << "[topo] Cycle detected!\n";
        return {};
    }

    return order;
}

} // namespace tcc
