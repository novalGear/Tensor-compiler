#include "graph.hpp"
#include <fstream>
#include <algorithm>

namespace tcc {

// ============================================================================
// Реализация load_from_proto для узлов (Парсинг атрибутов)
// полностью нейронка писала
// ============================================================================


// Хелпер для безопасного чтения булевых флагов из INT или BOOL
static bool get_bool_attr(const onnx::NodeProto& proto, const std::string& name, bool def = false) {
    for (const auto& attr : proto.attribute()) {
        if (attr.name() == name) {
            return attr.has_i() ? (attr.i() != 0) : def;
        }
    }
    return def;
}

// Хелпер для векторов int
static std::vector<int64_t> get_ints_attr(const onnx::NodeProto& proto, const std::string& name) {
    for (const auto& attr : proto.attribute()) {
        if (attr.name() == name) {
            if (attr.ints_size() > 0) {
                return std::vector<int64_t>(attr.ints().begin(), attr.ints().end());
            }
            return {};
        }
    }
    return {};
}

void AddNode::load_from_proto(const onnx::NodeProto& proto) {
    name = proto.name();
    input_tensors.assign(proto.input().begin(), proto.input().end());
    output_tensors.assign(proto.output().begin(), proto.output().end());
}

void MulNode::load_from_proto(const onnx::NodeProto& proto) {
    name = proto.name();
    input_tensors.assign(proto.input().begin(), proto.input().end());
    output_tensors.assign(proto.output().begin(), proto.output().end());
}

void ReluNode::load_from_proto(const onnx::NodeProto& proto) {
    name = proto.name();
    input_tensors.assign(proto.input().begin(), proto.input().end());
    output_tensors.assign(proto.output().begin(), proto.output().end());
}

void MatmulNode::load_from_proto(const onnx::NodeProto& proto) {
    name = proto.name();
    input_tensors.assign(proto.input().begin(), proto.input().end());
    output_tensors.assign(proto.output().begin(), proto.output().end());
}

void GemmNode::load_from_proto(const onnx::NodeProto& proto) {
    name = proto.name();
    input_tensors.assign(proto.input().begin(), proto.input().end());
    output_tensors.assign(proto.output().begin(), proto.output().end());

    transposeA = get_bool_attr(proto, "transA", false);
    transposeB = get_bool_attr(proto, "transB", false);
    // alpha/beta можно добавить при необходимости
}

void ConvNode::load_from_proto(const onnx::NodeProto& proto) {
    name = proto.name();
    input_tensors.assign(proto.input().begin(), proto.input().end());
    output_tensors.assign(proto.output().begin(), proto.output().end());

    strides = get_ints_attr(proto, "strides");
    pads = get_ints_attr(proto, "pads");
    group = get_bool_attr(proto, "group", 1) ? 1 : 1; // Упрощенно, лучше через get_int
    // Исправление для group (он обычно int):
    for (const auto& attr : proto.attribute()) {
        if (attr.name() == "group") group = attr.i();
    }
}

} // namespace tcc
