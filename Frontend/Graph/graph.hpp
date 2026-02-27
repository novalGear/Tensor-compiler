#pragma once

#include "onnx/onnx.pb.h"

#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>
#include <string>
#include <optional>
#include <iostream>
#include <functional>
#include <limits>

namespace tcc {

using TensorID = std::string;
using NodeID = size_t;
constexpr NodeID NO_PRODUCER = std::numeric_limits<NodeID>::max();
//
// struct Attribute {
//     std::string name;
//     // Используем variant для хранения разных типов значений
//     using Value = std::variant<bool, float, int64_t, std::string, std::vector<float>, std::vector<int64_t>>;
//     Value value;
// };

class ComputeNodeBase {
public:
    std::string             name;
    std::vector<TensorID>   input_tensors;
    std::vector<TensorID>   output_tensors;
    // std::unordered_map<std::string, Attribute> attributes;

    ComputeNodeBase() = default;
    explicit ComputeNodeBase(const std::string& n) : name(n) {}
    virtual ~ComputeNodeBase() = default;

    // Метод для конвертации узлов из Protobuf, наследники реализуют сами
    virtual void load_from_proto(const onnx::NodeProto& proto) = 0;
};

// ================================================================================================//
// Дочерние классы
// ================================================================================================//

class AddNode final : public ComputeNodeBase {
public:
    using ComputeNodeBase::ComputeNodeBase;
    void load_from_proto(const onnx::NodeProto& proto) override;
};


class MulNode final : public ComputeNodeBase {
public:
    using ComputeNodeBase::ComputeNodeBase;
    void load_from_proto(const onnx::NodeProto& proto) override;
};


class MatmulNode final : public ComputeNodeBase {
public:
    using ComputeNodeBase::ComputeNodeBase;
    void load_from_proto(const onnx::NodeProto& proto) override;
};


class ReluNode final : public ComputeNodeBase {
public:
    using ComputeNodeBase::ComputeNodeBase;
    void load_from_proto(const onnx::NodeProto& proto) override;
};

class ConvNode final : public ComputeNodeBase {
public:
    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::vector<int64_t> dilations;
    int64_t group = 1;

    using ComputeNodeBase::ComputeNodeBase;
    void load_from_proto(const onnx::NodeProto& proto) override;
};

class GemmNode final : public ComputeNodeBase {
public:
    float alpha = 1.0f;
    float beta = 1.0f;

    bool transposeA = false;
    bool transposeB = false;

    using ComputeNodeBase::ComputeNodeBase;
    void load_from_proto(const onnx::NodeProto& proto) override;
};
// ====================================================================================================================//

using ComputeNode = std::variant<
    AddNode,
    MulNode,
    MatmulNode,
    GemmNode,
    ReluNode,
    ConvNode
>;

struct TensorDescription {
public:
    std::vector<size_t> dimensions;
    NodeID              producer_node_id = NO_PRODUCER;
    std::vector<NodeID> consumer_node_ids;
    bool is_graph_input = false;
};


class ComputeGraph {
public:
    // хранилище узлов как variant
    std::vector<ComputeNode> nodes;

    // Имя тензора -> его описание
    std::unordered_map<TensorID, TensorDescription> tensor_map;

    static std::unique_ptr<ComputeGraph> load_from_onnx(const std::string& filepath);

    // Методы для обхода (BFS)
    // template<typename Visitor>
    // void bfs_traverse(Visitor&& visitor) const;

private:

    // Этапы конвертации
    static bool read_from_onnx_proto(const std::string& filepath, onnx::ModelProto& model_out);
    static std::unique_ptr<ComputeGraph> convertion(const onnx::ModelProto& model_proto);

     // --- Приватные методы-помощники экземпляра
    void register_graph_inputs(const onnx::GraphProto& gp);
    void register_initializers(const onnx::GraphProto& gp);
    void build_nodes(const onnx::GraphProto& gp);
    void fill_tensor_shapes(const onnx::GraphProto& gp);
    void update_tensor_connections(NodeID node_id, const ComputeNode& node);
};

} // namespace tcc
