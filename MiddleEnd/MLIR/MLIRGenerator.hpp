// MLIRGenerator.hpp
#pragma once

#include "graph.hpp"
#include "OperationEmitters/IOperationEmitter.hpp"
#include "TypeConverter.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace tcc {
namespace mlir_gen {

// Forward declarations
class AddEmitter;
class MulEmitter;
class ConstantEmitter;
class ReLUEmitter;
class MatMulEmitter;

class MLIRGenerator {
public:
    struct Config {
        bool printMLIR = false;
        bool enableFusion = false;
        std::string outputFile = "";
    };

    explicit MLIRGenerator(const Config& cfg);
    MLIRGenerator();
    ~MLIRGenerator();

    bool generate(const ComputeGraph& graph);
    mlir::OwningOpRef<mlir::ModuleOp> takeModule();
    void printMLIRToStream(std::ostream& os);
    bool saveMLIRToFile(const std::string& filename);

private:
    void initMLIRContext();
    void initEmitters();
    std::vector<size_t> topologicalSort(const ComputeGraph& graph);
    std::vector<TensorID> collectGraphInputs(const ComputeGraph& graph);
    std::vector<TensorID> collectGraphOutputs(const ComputeGraph& graph);
    std::vector<size_t> getTensorDims(const ComputeGraph& graph, const TensorID& tensorId);
    bool createFunctionArguments(const std::vector<TensorID>& inputs,
                                 const std::vector<TensorID>& outputs);
    bool createFunctionReturn(const std::vector<TensorID>& outputs);
    bool createMainFunction(const ComputeGraph& graph);
    bool emitNode(const ComputeGraph& graph, size_t nodeId, const ComputeNode& node);

    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace mlir_gen
} // namespace tcc
