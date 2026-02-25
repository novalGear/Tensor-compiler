#pragma once

#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>
#include <string>

namespace tcc {

using TensorID = std::string;
using NodeID = size_t;

enum class NodeType {
    ADD,
    MUL,
    RELU,
    CONV,
    GEMM
};

class ComputeNodeBase {
    // ? NodeType type;

    std::vector<TensorID>  input_tensors;
    std::vector<TensorID> output_tensors;



};

class TensorDescription {
    std::vector<size_t> dimensions;
    NodeID              producer;
    std::vector<NodeID> users;
    //TODO: store tensor data somewhere
};


class AddNode final :  ComputeNodeBase {

};


class MulNode final :  ComputeNodeBase {

};


class MatmulNode final :  ComputeNodeBase {

};


class ReluNode final :  ComputeNodeBase {

};

class ConvNode final :  ComputeNodeBase {

};

class GemmNode final :  ComputeNodeBase {

    bool transposeA;
    bool transposeB;
};

using ComputeNode = std::variant<
    AddNode,
    MulNode,
    MatmulNode,
    GemmNode,
    ReluNode,
    ConvNode
>;



class ComputeGraph {
    std::vector<ComputeNode> nodes;

    std::unordered_map<TensorID, TensorDescription> tensor_map;

};

}
