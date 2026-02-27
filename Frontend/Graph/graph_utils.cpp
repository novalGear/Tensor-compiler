#include "graph.hpp"

#include <string>

// std::string get_type_name(const ComputeNode& node) {
//     return std::visit([](const auto& n) -> std::string {
//         using T = std::decay_t<decltype(n)>;
//         if constexpr (std::is_same_v<T, AddNode>) return "Add";
//         if constexpr (std::is_same_v<T, MulNode>) return "Mul";
//         if constexpr (std::is_same_v<T, MatmulNode>) return "MatMul";
//         if constexpr (std::is_same_v<T, GemmNode>) return "Gemm";
//         if constexpr (std::is_same_v<T, ReluNode>) return "Relu";
//         if constexpr (std::is_same_v<T, ConvNode>) return "Conv";
//         return "Unknown";
//     }, node);
// }
