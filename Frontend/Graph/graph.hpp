#pragma once

#include "onnx/onnx-ml.pb.h"
// Подключаем сгенерированные структуры и variant
#include "graph_gen.hpp"

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <limits>
#include <iostream>

namespace tcc {

using TensorID = std::string;
using NodeID = size_t;

constexpr NodeID NO_PRODUCER = std::numeric_limits<NodeID>::max();

// Описание тензора
struct TensorDescription {
    std::vector<size_t> dimensions;
    NodeID producer_node_id = NO_PRODUCER;
    std::vector<NodeID> consumer_node_ids;
    bool is_graph_input = false;
    bool is_initializer = false;
};

class ComputeGraph {
public:
    // Хранилище узлов (variant из сгенерированных типов)
    std::vector<ComputeNode> nodes;

    // Карта тензоров
    std::unordered_map<TensorID, TensorDescription> tensor_descr_map;

    NodeID addNode(ComputeNode&& node);
    static ConstantNode makeConstantNode(const std::string& tensor_name,
                                          std::vector<float> data);

    static std::unique_ptr<ComputeGraph> load_from_onnx(const std::string& filepath);

    std::vector<size_t> topologicalSort(bool verbose) const;

    std::vector<TensorID> collectInputs() const;
    std::vector<TensorID> collectOutputs() const;

    std::vector<size_t> getTensorDims(const TensorID& tensorId) const;

    void print_tensor_descr_map(std::ostream& os = std::cout) const;
    void print_nodes(std::ostream& os = std::cout) const;
    void print_inputs_outputs(std::ostream& os = std::cout) const;
    void print_graph_info(std::ostream& os = std::cout) const;

private:
    static bool read_from_onnx_proto(const std::string& filepath, onnx::ModelProto& model_out);
    static std::unique_ptr<ComputeGraph> convertion(const onnx::ModelProto& model_proto);

    void register_graph_inputs(const onnx::GraphProto& gp);
    void register_initializers(const onnx::GraphProto& gp);
    void build_nodes(const onnx::GraphProto& gp);
    void update_tensor_connections(NodeID node_id, const ComputeNode& node);
    void fill_tensor_shapes(const onnx::GraphProto& gp);


    void create_constant_nodes_from_initializers(const onnx::GraphProto& gp);
    static std::vector<float> extractInitializerData(const onnx::TensorProto& initializer);
    void updateTensorMapForConstant(const std::string& tensor_name,
                                     NodeID node_id,
                                     const onnx::TensorProto& initializer);

    static std::string getNodeTypeName(const ComputeNode& node);

    void inferAllTensorShapes();

    // void inferNodeShapes();
    void inferMatMulShape(const MatmulNode& node, const TensorID& outputTensor);
    void inferAddShape(const AddNode& node, const TensorID& outputTensor);
    void inferMulShape(const MulNode& node, const TensorID& outputTensor);
    void inferReluShape(const ReluNode& node, const TensorID& outputTensor);
    void inferGemmShape(const GemmNode& node, const TensorID& outputTensor);
    void inferConvShape(const ConvNode& node, const TensorID& outputTensor);
    void inferFlattenShape(const FlattenNode& node, const TensorID& outputTensor);

    std::vector<size_t> broadcastDims(const std::vector<size_t>& a, const std::vector<size_t>& b);
    void setTensorDims(const TensorID& tensor, const std::vector<size_t>& dims);
    bool needsShapeInference(const TensorID& tensor) const;
};

} // namespace tcc
