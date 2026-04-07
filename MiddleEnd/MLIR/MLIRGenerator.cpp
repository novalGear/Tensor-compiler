// MLIRGenerator.cpp - переписанный

#include "MLIRGenerator.hpp"
#include "TypeConverter.hpp"
#include "OperationEmitters/AddEmitter.hpp"
#include "OperationEmitters/MulEmitter.hpp"
#include "OperationEmitters/ConstantEmitter.hpp"
#include "OperationEmitters/ReluEmitter.hpp"
#include "OperationEmitters/MatMulEmitter.hpp"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <fstream>
#include <iostream>

namespace tcc {
namespace mlir_gen {

//==============================================================================
// IMPL STRUCT
//==============================================================================

struct MLIRGenerator::Impl {
    mlir::MLIRContext context;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::OpBuilder builder;
    mlir::func::FuncOp mainFunc;
    std::unordered_map<TensorID, mlir::Value> tensorMap;
    std::unique_ptr<TypeConverter> typeConverter;
    Config config;

    // Эмиттеры
    std::unique_ptr<AddEmitter> addEmitter;
    std::unique_ptr<MulEmitter> mulEmitter;
    std::unique_ptr<ConstantEmitter> constantEmitter;
    std::unique_ptr<ReluEmitter> reluEmitter;
    std::unique_ptr<MatMulEmitter> matmulEmitter;

    Impl(const Config& cfg) : config(cfg), builder(&context) {
        loadDialects();
        module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
        builder = mlir::OpBuilder(module->getBodyRegion());
        typeConverter = std::make_unique<TypeConverter>(&context);
    }

private:
    void loadDialects() {
        context.getOrLoadDialect<mlir::BuiltinDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
        context.getOrLoadDialect<mlir::tensor::TensorDialect>();
    }
};

//==============================================================================
// CONSTRUCTORS / DESTRUCTOR
//==============================================================================

MLIRGenerator::MLIRGenerator(const Config& cfg)
    : pImpl(std::make_unique<Impl>(cfg)) {
    initEmitters();
}

MLIRGenerator::MLIRGenerator() : MLIRGenerator(Config()) {}
MLIRGenerator::~MLIRGenerator() = default;

void MLIRGenerator::initEmitters() {
    pImpl->addEmitter = std::make_unique<AddEmitter>(pImpl->builder, pImpl->tensorMap);
    pImpl->mulEmitter = std::make_unique<MulEmitter>(pImpl->builder, pImpl->tensorMap);
    pImpl->constantEmitter = std::make_unique<ConstantEmitter>(pImpl->builder, pImpl->tensorMap);
    pImpl->reluEmitter = std::make_unique<ReluEmitter>(pImpl->builder, pImpl->tensorMap);
    pImpl->matmulEmitter = std::make_unique<MatMulEmitter>(pImpl->builder, pImpl->tensorMap);
}

//==============================================================================
// DEBUG HELPERS
//==============================================================================

void MLIRGenerator::debugPrintTensorMap(const std::string& phase) {
    std::cout << "[DEBUG] TensorMap " << phase << " (" << pImpl->tensorMap.size() << " entries):\n";
    for (const auto& [name, val] : pImpl->tensorMap) {
        std::cout << "  '" << name << "'\n";
    }
}

void MLIRGenerator::debugPrintNode(size_t nodeId, const ComputeNode& node) {
    std::visit([nodeId](const auto& n) {
        std::cout << "[DEBUG] Node " << nodeId << " (" << n.getTypeName() << ")\n";
        std::cout << "  Inputs: ";
        for (const auto& in : n.input_tensors) std::cout << "'" << in << "' ";
        std::cout << "\n  Outputs: ";
        for (const auto& out : n.output_tensors) std::cout << "'" << out << "' ";
        std::cout << "\n";
    }, node);
}

//==============================================================================
// FUNCTION CREATION
//==============================================================================

bool MLIRGenerator::createMainFunction(const ComputeGraph& graph) {
    std::cout << "[DEBUG] === createMainFunction START ===\n";

    auto inputs = graph.collectInputs();
    if (inputs.empty()) {
        std::cerr << "[ERROR] No inputs found in graph!\n";
        return false;
    }

    auto outputs = graph.collectOutputs();
    auto loc = mlir::UnknownLoc::get(&pImpl->context);

    // Создаём типы
    auto argTypes = createArgumentTypes(graph, inputs);
    auto returnTypes = createReturnTypes(graph, outputs);

    // Создаём функцию
    auto funcType = mlir::FunctionType::get(&pImpl->context, argTypes, returnTypes);
    pImpl->mainFunc = pImpl->builder.create<mlir::func::FuncOp>(loc, "forward", funcType);
    pImpl->mainFunc.setPrivate();
    pImpl->mainFunc.addEntryBlock();
    pImpl->builder.setInsertionPointToStart(&pImpl->mainFunc.getBody().front());

    // Маппим входные аргументы
    for (size_t i = 0; i < inputs.size(); ++i) {
        pImpl->tensorMap[inputs[i]] = pImpl->mainFunc.getArgument(i);
        std::cout << "[DEBUG]   Mapped input: '" << inputs[i] << "' -> arg" << i << "\n";
    }

    debugPrintTensorMap("after inputs");
    return true;
}

llvm::SmallVector<mlir::Type> MLIRGenerator::createArgumentTypes(
    const ComputeGraph& graph, const std::vector<TensorID>& inputs) {

    llvm::SmallVector<mlir::Type> types;
    for (const auto& inputName : inputs) {
        auto dims = graph.getTensorDims(inputName);
        types.push_back(pImpl->typeConverter->toTensorType(dims));
    }
    return types;
}

llvm::SmallVector<mlir::Type> MLIRGenerator::createReturnTypes(
    const ComputeGraph& graph, const std::vector<TensorID>& outputs) {

    llvm::SmallVector<mlir::Type> types;
    for (const auto& outputName : outputs) {
        auto dims = graph.getTensorDims(outputName);
        types.push_back(pImpl->typeConverter->toTensorType(dims));
    }
    return types;
}

bool MLIRGenerator::createFunctionReturn(const std::vector<TensorID>& outputs) {
    auto loc = mlir::UnknownLoc::get(&pImpl->context);

    llvm::SmallVector<mlir::Value> returnValues;
    for (const auto& outputName : outputs) {
        auto it = pImpl->tensorMap.find(outputName);
        if (it == pImpl->tensorMap.end()) {
            std::cerr << "[ERROR] Output tensor '" << outputName << "' not found in tensorMap\n";
            return false;
        }
        returnValues.push_back(it->second);
    }

    pImpl->builder.create<mlir::func::ReturnOp>(loc, returnValues);
    std::cout << "[DEBUG] Created return with " << returnValues.size() << " values\n";

    return true;
}

//==============================================================================
// NODE EMISSION
//==============================================================================

bool MLIRGenerator::emitNode(const ComputeGraph& graph, size_t nodeId, const ComputeNode& node) {
    return std::visit([this, &graph, nodeId](const auto& n) -> bool {
        using T = std::decay_t<decltype(n)>;

        // Сбор входных значений
        std::vector<mlir::Value> inputs;
        for (const auto& inputName : n.input_tensors) {
            auto it = pImpl->tensorMap.find(inputName);
            if (it == pImpl->tensorMap.end()) {
                std::cerr << "Error: Input tensor " << inputName << " not found\n";
                return false;
            }
            inputs.push_back(it->second);
        }

        // Получение размерностей выхода
        std::vector<size_t> outputDims;
        if (!n.output_tensors.empty()) {
            outputDims = graph.getTensorDims(n.output_tensors[0]);
        }

        // Эмиссия в зависимости от типа
        if constexpr (std::is_same_v<T, ConstantNode>) {
            std::cout << "[DEBUG] Emitting ConstantNode\n";
            pImpl->constantEmitter->emitConstant(n.value, n.output_tensors, outputDims);
        }
        else if constexpr (std::is_same_v<T, AddNode>) {
            std::cout << "[DEBUG] Emitting AddNode\n";
            if (inputs.size() != 2) return false;
            pImpl->addEmitter->emit(inputs, n.output_tensors, outputDims);
        }
        else if constexpr (std::is_same_v<T, MulNode>) {
            std::cout << "[DEBUG] Emitting MulNode\n";
            if (inputs.size() != 2) return false;
            pImpl->mulEmitter->emit(inputs, n.output_tensors, outputDims);
        }
        else if constexpr (std::is_same_v<T, ReluNode>) {
            std::cout << "[DEBUG] Emitting ReluNode\n";
            if (inputs.size() != 1) return false;
            pImpl->reluEmitter->emit(inputs, n.output_tensors, outputDims);
        }
        else if constexpr (std::is_same_v<T, MatmulNode>) {
            std::cout << "[DEBUG] Emitting MatmulNode\n";
            if (inputs.size() != 2) return false;
            pImpl->matmulEmitter->emit(inputs, n.output_tensors, outputDims);
        }
        else {
            std::cerr << "Error: Unsupported node type\n";
            return false;
        }

        return true;
    }, node);
}

bool MLIRGenerator::validateInputs(const std::vector<TensorID>& inputTensors) {
    for (const auto& inputName : inputTensors) {
        if (pImpl->tensorMap.find(inputName) == pImpl->tensorMap.end()) {
            std::cerr << "[ERROR] Input tensor '" << inputName << "' not found in tensorMap\n";
            debugPrintTensorMap("before error");
            return false;
        }
    }
    return true;
}

std::vector<mlir::Value> MLIRGenerator::collectInputValues(const std::vector<TensorID>& inputTensors) {
    std::vector<mlir::Value> inputs;
    inputs.reserve(inputTensors.size());
    for (const auto& inputName : inputTensors) {
        inputs.push_back(pImpl->tensorMap[inputName]);
    }
    return inputs;
}

std::vector<std::vector<size_t>> MLIRGenerator::collectOutputDims(
    const ComputeGraph& graph, const std::vector<TensorID>& outputTensors) {

    std::vector<std::vector<size_t>> dimsList;
    for (const auto& outputName : outputTensors) {
        dimsList.push_back(graph.getTensorDims(outputName));
    }
    return dimsList;
}

//==============================================================================
// MAIN GENERATION
//==============================================================================

bool MLIRGenerator::generate(const ComputeGraph& graph) {
    std::cout << "\n============================================================\n";
    std::cout << "MLIR GENERATION START\n";
    std::cout << "============================================================\n\n";

    if (!createMainFunction(graph)) {
        std::cerr << "[ERROR] Failed to create main function\n";
        return false;
    }

    auto order = graph.topologicalSort(true);
    std::cout << "[DEBUG] Topological order: ";
    for (auto id : order) std::cout << id << " ";
    std::cout << "\n\n";

    for (size_t nodeId : order) {
        if (!emitNode(graph, nodeId, graph.nodes[nodeId])) {
            std::cerr << "[ERROR] Failed to emit node " << nodeId << "\n";
            return false;
        }
        debugPrintTensorMap("after node " + std::to_string(nodeId));
    }

    auto outputs = graph.collectOutputs();
    if (!createFunctionReturn(outputs)) {
        std::cerr << "[ERROR] Failed to create function return\n";
        return false;
    }

    if (mlir::failed(mlir::verify(*pImpl->module))) {
        std::cerr << "[ERROR] MLIR verification failed\n";
        return false;
    }

    std::cout << "\n============================================================\n";
    std::cout << "MLIR GENERATION SUCCESS\n";
    std::cout << "============================================================\n";

    return true;
}

//==============================================================================
// OUTPUT
//==============================================================================

mlir::OwningOpRef<mlir::ModuleOp> MLIRGenerator::takeModule() {
    return std::move(pImpl->module);
}

void MLIRGenerator::printMLIRToStream(std::ostream& os) {
    if (pImpl->module) {
        std::string str;
        llvm::raw_string_ostream rss(str);
        pImpl->module->print(rss);
        os << str;
    }
}

bool MLIRGenerator::saveMLIRToFile(const std::string& filename) {
    std::error_code ec;
    llvm::raw_fd_ostream os(filename, ec);
    if (ec) {
        std::cerr << "Cannot open file: " << filename << "\n";
        return false;
    }
    pImpl->module->print(os);
    os.close();
    std::cout << "MLIR saved to: " << filename << "\n";
    return true;
}

} // namespace mlir_gen
} // namespace tcc
