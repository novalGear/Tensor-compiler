// MLIRGenerator.cpp
#include "MLIRGenerator.hpp"
#include "TypeConverter.hpp"
#include "OperationEmitters/AddEmitter.hpp"
#include "OperationEmitters/MulEmitter.hpp"
#include "OperationEmitters/ConstantEmitter.hpp"
#include "OperationEmitters/ReLUEmitter.hpp"
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
#include <queue>
#include <set>
#include <algorithm>

namespace tcc {
namespace mlir_gen {

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
    std::unique_ptr<ReLUEmitter> reluEmitter;
    std::unique_ptr<MatMulEmitter> matmulEmitter;

    Impl(const Config& cfg) : config(cfg), builder(&context) {
        typeConverter = std::make_unique<TypeConverter>(&context);
    }
};

MLIRGenerator::MLIRGenerator(const Config& cfg)
    : pImpl(std::make_unique<Impl>(cfg)) {
    initMLIRContext();
    initEmitters();
}

// Конструктор без параметров (использует Config по умолчанию)
MLIRGenerator::MLIRGenerator()
    : MLIRGenerator(Config()) {}

MLIRGenerator::~MLIRGenerator() = default;

void MLIRGenerator::initMLIRContext() {
    pImpl->context.getOrLoadDialect<mlir::BuiltinDialect>();
    pImpl->context.getOrLoadDialect<mlir::func::FuncDialect>();
    pImpl->context.getOrLoadDialect<mlir::arith::ArithDialect>();
    pImpl->context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
    pImpl->context.getOrLoadDialect<mlir::tensor::TensorDialect>();

    pImpl->module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&pImpl->context));
    pImpl->builder = mlir::OpBuilder(pImpl->module->getBodyRegion());
}

void MLIRGenerator::initEmitters() {
    pImpl->addEmitter = std::make_unique<AddEmitter>(pImpl->builder, pImpl->tensorMap);
    pImpl->mulEmitter = std::make_unique<MulEmitter>(pImpl->builder, pImpl->tensorMap);
    pImpl->constantEmitter = std::make_unique<ConstantEmitter>(pImpl->builder, pImpl->tensorMap);
    pImpl->reluEmitter = std::make_unique<ReLUEmitter>(pImpl->builder, pImpl->tensorMap);
    pImpl->matmulEmitter = std::make_unique<MatMulEmitter>(pImpl->builder, pImpl->tensorMap);
}

std::vector<size_t> MLIRGenerator::topologicalSort(const ComputeGraph& graph) {
    std::vector<std::vector<size_t>> dependencies(graph.nodes.size());
    std::vector<int> inDegree(graph.nodes.size(), 0);

    // Строим граф зависимостей
    for (const auto& [tensorId, tensorDesc] : graph.tensor_map) {
        if (tensorDesc.producer_node_id != NO_PRODUCER) {
            for (NodeID consumerId : tensorDesc.consumer_node_ids) {
                dependencies[tensorDesc.producer_node_id].push_back(consumerId);
                inDegree[consumerId]++;
            }
        }
    }

    // Алгоритм Кана
    std::queue<size_t> queue;
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        if (inDegree[i] == 0) {
            queue.push(i);
        }
    }

    std::vector<size_t> order;
    while (!queue.empty()) {
        size_t nodeId = queue.front();
        queue.pop();
        order.push_back(nodeId);

        for (size_t dependentId : dependencies[nodeId]) {
            inDegree[dependentId]--;
            if (inDegree[dependentId] == 0) {
                queue.push(dependentId);
            }
        }
    }

    if (order.size() != graph.nodes.size()) {
        std::cerr << "Error: Graph contains a cycle!\n";
        return {};
    }

    return order;
}

std::vector<TensorID> MLIRGenerator::collectGraphInputs(const ComputeGraph& graph) {
    std::vector<TensorID> inputs;
    for (const auto& [tensorId, tensorDesc] : graph.tensor_map) {
        if (tensorDesc.is_graph_input && !tensorDesc.is_initializer) {
            inputs.push_back(tensorId);
        }
    }
    return inputs;
}

std::vector<TensorID> MLIRGenerator::collectGraphOutputs(const ComputeGraph& graph) {
    std::vector<TensorID> outputs;
    for (const auto& [tensorId, tensorDesc] : graph.tensor_map) {
        if (tensorDesc.consumer_node_ids.empty() &&
            !tensorDesc.is_graph_input &&
            tensorDesc.producer_node_id != NO_PRODUCER) {
            outputs.push_back(tensorId);
        }
    }
    return outputs;
}

std::vector<size_t> MLIRGenerator::getTensorDims(const ComputeGraph& graph, const TensorID& tensorId) {
    auto it = graph.tensor_map.find(tensorId);
    if (it != graph.tensor_map.end()) {
        return it->second.dimensions;
    }
    return {};
}

bool MLIRGenerator::createFunctionArguments(const std::vector<TensorID>& inputs) {
    auto loc = mlir::UnknownLoc::get(&pImpl->context);

    llvm::SmallVector<mlir::Type> argTypes;
    for (const auto& inputName : inputs) {
        // Пока используем динамические размерности
        auto tensorType = mlir::RankedTensorType::get(
            {mlir::ShapedType::kDynamic},
            pImpl->typeConverter->getElementType());
        argTypes.push_back(tensorType);
    }

    auto funcType = mlir::FunctionType::get(&pImpl->context, argTypes, {});

    pImpl->mainFunc = pImpl->builder.create<mlir::func::FuncOp>(loc, "forward", funcType);
    pImpl->mainFunc.setPrivate();

    mlir::Region& region = pImpl->mainFunc.getBody();
    region.push_back(new mlir::Block);
    mlir::Block& block = region.front();

    pImpl->builder.setInsertionPointToStart(&block);

    // Добавляем аргументы в карту тензоров
    for (size_t i = 0; i < inputs.size(); ++i) {
        mlir::Value arg = block.getArgument(i);
        pImpl->tensorMap[inputs[i]] = arg;
    }

    return true;
}

bool MLIRGenerator::createFunctionReturn(const std::vector<TensorID>& outputs) {
    auto loc = mlir::UnknownLoc::get(&pImpl->context);

    llvm::SmallVector<mlir::Value> returnValues;
    for (const auto& outputName : outputs) {
        auto it = pImpl->tensorMap.find(outputName);
        if (it == pImpl->tensorMap.end()) {
            std::cerr << "Error: Output tensor " << outputName << " not found\n";
            return false;
        }
        returnValues.push_back(it->second);
    }

    pImpl->builder.create<mlir::func::ReturnOp>(loc, returnValues);

    // Обновляем тип функции с учетом возвращаемых значений
    auto funcType = mlir::FunctionType::get(&pImpl->context,
        pImpl->mainFunc.getFunctionType().getInputs(),
        llvm::SmallVector<mlir::Type>(returnValues.size(),
            pImpl->typeConverter->getElementType()));
    pImpl->mainFunc.setType(funcType);

    return true;
}

bool MLIRGenerator::createMainFunction(const ComputeGraph& graph) {
    auto inputs = collectGraphInputs(graph);
    auto outputs = collectGraphOutputs(graph);

    if (!createFunctionArguments(inputs)) {
        return false;
    }

    // Эмиттеры будут вставлены здесь

    if (!createFunctionReturn(outputs)) {
        return false;
    }

    return true;
}

bool MLIRGenerator::emitNode(const ComputeGraph& graph, size_t nodeId, const ComputeNode& node) {
    return std::visit([this, &graph, nodeId](const auto& n) -> bool {
        using T = std::decay_t<decltype(n)>;

        std::vector<mlir::Value> inputs;
        for (const auto& inputName : n.input_tensors) {
            auto it = pImpl->tensorMap.find(inputName);
            if (it == pImpl->tensorMap.end()) {
                std::cerr << "Error: Input tensor " << inputName << " not found\n";
                return false;
            }
            inputs.push_back(it->second);
        }

        std::vector<std::vector<size_t>> outputDimsList;
        for (const auto& outputName : n.output_tensors) {
            outputDimsList.push_back(getTensorDims(graph, outputName));
        }

        if constexpr (std::is_same_v<T, AddNode>) {
            if (inputs.size() != 2 || outputDimsList.empty()) return false;
            pImpl->addEmitter->emit(inputs, n.output_tensors, outputDimsList[0]);
        }
        else if constexpr (std::is_same_v<T, MulNode>) {
            if (inputs.size() != 2 || outputDimsList.empty()) return false;
            pImpl->mulEmitter->emit(inputs, n.output_tensors, outputDimsList[0]);
        }
        else if constexpr (std::is_same_v<T, ConstantNode>) {
            if (outputDimsList.empty()) return false;
            pImpl->constantEmitter->emit({}, n.output_tensors, outputDimsList[0]);
        }
        else if constexpr (std::is_same_v<T, ReluNode>) {
            if (inputs.empty() || outputDimsList.empty()) return false;
            pImpl->reluEmitter->emit(inputs, n.output_tensors, outputDimsList[0]);
        }
        else if constexpr (std::is_same_v<T, MatmulNode>) {
            if (inputs.size() != 2 || outputDimsList.empty()) return false;
            pImpl->matmulEmitter->emit(inputs, n.output_tensors, outputDimsList[0]);
        }
        else {
            std::cerr << "Error: Unsupported node type\n";
            return false;
        }

        return true;
    }, node);
}

bool MLIRGenerator::generate(const ComputeGraph& graph) {
    auto order = topologicalSort(graph);
    if (order.empty()) {
        std::cerr << "Error: Topological sort failed\n";
        return false;
    }

    if (!createMainFunction(graph)) {
        std::cerr << "Error: Failed to create main function\n";
        return false;
    }

    for (size_t nodeId : order) {
        if (!emitNode(graph, nodeId, graph.nodes[nodeId])) {
            std::cerr << "Error: Failed to emit node " << nodeId << "\n";
            return false;
        }
    }

    if (mlir::failed(mlir::verify(*pImpl->module))) {
        std::cerr << "Error: MLIR module verification failed\n";
        return false;
    }

    if (pImpl->config.printMLIR) {
        printMLIRToStream(std::cout);
    }

    if (!pImpl->config.outputFile.empty()) {
        saveMLIRToFile(pImpl->config.outputFile);
    }

    return true;
}

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
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Cannot open file: " << filename << "\n";
        return false;
    }
    printMLIRToStream(out);
    out.close();
    std::cout << "MLIR saved to: " << filename << "\n";
    return true;
}

} // namespace mlir_gen
} // namespace tcc
