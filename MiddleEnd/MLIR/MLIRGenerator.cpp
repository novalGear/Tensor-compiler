// MLIRGenerator.cpp


#include "../Pipeline/MLIRPasses.hpp"
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

#include "graph.hpp"

namespace tcc {
namespace mlir_gen {

struct MLIRGenerator::Impl {
    mlir::MLIRContext context;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::OpBuilder builder;
    mlir::func::FuncOp mainFunc;
    std::unordered_map<TensorID, mlir::Value> tensorMap;
    std::unique_ptr<TypeConverter> typeConverter;
    const ComputeGraph* currentGraph = nullptr;
    Config config;

    // Эмиттеры
    std::unique_ptr<AddEmitter> addEmitter;
    std::unique_ptr<MulEmitter> mulEmitter;
    std::unique_ptr<ConstantEmitter> constantEmitter;
    std::unique_ptr<ReLUEmitter> reluEmitter;
    std::unique_ptr<MatMulEmitter> matmulEmitter;

    Impl(const Config& cfg) : config(cfg), builder(&context) {
        context.getOrLoadDialect<mlir::BuiltinDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
        context.getOrLoadDialect<mlir::tensor::TensorDialect>();

        module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
        builder = mlir::OpBuilder(module->getBodyRegion());
        typeConverter = std::make_unique<TypeConverter>(&context);
    }
};

MLIRGenerator::MLIRGenerator(const Config& cfg)
    : pImpl(std::make_unique<Impl>(cfg)) {
    initMLIRContext();
    initEmitters();
}

MLIRGenerator::MLIRGenerator() : MLIRGenerator(Config()) {}
MLIRGenerator::~MLIRGenerator() = default;

void MLIRGenerator::initMLIRContext() {
    // Диалекты уже загружены в конструкторе Impl
}

void MLIRGenerator::initEmitters() {
    pImpl->addEmitter = std::make_unique<AddEmitter>(pImpl->builder, pImpl->tensorMap);
    pImpl->mulEmitter = std::make_unique<MulEmitter>(pImpl->builder, pImpl->tensorMap);
    pImpl->constantEmitter = std::make_unique<ConstantEmitter>(pImpl->builder, pImpl->tensorMap);
    pImpl->reluEmitter = std::make_unique<ReLUEmitter>(pImpl->builder, pImpl->tensorMap);
    pImpl->matmulEmitter = std::make_unique<MatMulEmitter>(pImpl->builder, pImpl->tensorMap);
}

bool MLIRGenerator::createFunctionArguments(const std::vector<TensorID>& inputs,
                                             const std::vector<TensorID>& outputs) {
    auto loc = mlir::UnknownLoc::get(&pImpl->context);

    llvm::SmallVector<mlir::Type> argTypes;
    for (const auto& inputName : inputs) {
        auto dims = pImpl->currentGraph->getTensorDims(inputName);  // <-- исправлено
        auto tensorType = pImpl->typeConverter->toTensorType(dims);
        argTypes.push_back(tensorType);
    }

    llvm::SmallVector<mlir::Type> returnTypes;
    for (const auto& outputName : outputs) {
        auto dims = pImpl->currentGraph->getTensorDims(outputName);  // <-- исправлено
        auto tensorType = pImpl->typeConverter->toTensorType(dims);
        returnTypes.push_back(tensorType);
    }

    auto funcType = mlir::FunctionType::get(&pImpl->context, argTypes, returnTypes);

    pImpl->mainFunc = pImpl->builder.create<mlir::func::FuncOp>(loc, "forward", funcType);
    pImpl->mainFunc.setPrivate();

    mlir::Region& region = pImpl->mainFunc.getBody();
    region.push_back(new mlir::Block);
    mlir::Block& block = region.front();

    pImpl->builder.setInsertionPointToStart(&block);

    for (size_t i = 0; i < inputs.size(); ++i) {
        mlir::Value arg = block.getArgument(i);
        pImpl->tensorMap[inputs[i]] = arg;
    }

    return true;
}

bool MLIRGenerator::createFunctionReturn(const std::vector<TensorID>& outputs) {
    std::cout << "[DEBUG] createFunctionReturn START, outputs=" << outputs.size() << "\n";

    // Выводим содержимое tensorMap для отладки
    std::cout << "[DEBUG] tensorMap contents:\n";
    for (const auto& [name, val] : pImpl->tensorMap) {
        std::cout << "    " << name << "\n";
    }

    auto loc = mlir::UnknownLoc::get(&pImpl->context);

    llvm::SmallVector<mlir::Value> returnValues;
    for (const auto& outputName : outputs) {
        auto it = pImpl->tensorMap.find(outputName);
        if (it == pImpl->tensorMap.end()) {
            std::cerr << "Error: Output tensor " << outputName << " not found in tensorMap\n";
            return false;
        }
        returnValues.push_back(it->second);
        std::cout << "[DEBUG]   returning " << outputName << "\n";
    }

    pImpl->builder.create<mlir::func::ReturnOp>(loc, returnValues);

    std::cout << "[DEBUG] createFunctionReturn END\n";
    return true;
}

bool MLIRGenerator::createMainFunction(const ComputeGraph& graph) {
    std::cout << "[DEBUG] createMainFunction START\n";

    auto inputs = graph.collectInputs();
    auto outputs = graph.collectOutputs();

    std::cout << "[DEBUG] inputs: " << inputs.size() << ", outputs: " << outputs.size() << "\n";

    auto loc = mlir::UnknownLoc::get(&pImpl->context);

    // Создаем типы аргументов
    llvm::SmallVector<mlir::Type> argTypes;
    for (const auto& inputName : inputs) {
        auto dims = graph.getTensorDims(inputName);
        auto tensorType = pImpl->typeConverter->toTensorType(dims);
        argTypes.push_back(tensorType);
        std::cout << "[DEBUG] Input " << inputName << " dims: " << dims.size() << "\n";
    }

    // Создаем типы возвращаемых значений
    llvm::SmallVector<mlir::Type> returnTypes;
    for (const auto& outputName : outputs) {
        auto dims = graph.getTensorDims(outputName);
        auto tensorType = pImpl->typeConverter->toTensorType(dims);
        returnTypes.push_back(tensorType);
        std::cout << "[DEBUG] Output " << outputName << " dims: " << dims.size() << "\n";
    }

    auto funcType = mlir::FunctionType::get(&pImpl->context, argTypes, returnTypes);

    std::cout << "[DEBUG] Creating FuncOp...\n";
    pImpl->mainFunc = pImpl->builder.create<mlir::func::FuncOp>(loc, "forward", funcType);
    pImpl->mainFunc.setPrivate();

    std::cout << "[DEBUG] Adding entry block...\n";
    pImpl->mainFunc.addEntryBlock();

    std::cout << "[DEBUG] Setting insertion point...\n";
    pImpl->builder.setInsertionPointToStart(&pImpl->mainFunc.getBody().front());

    // ВАЖНО: Добавляем входные тензоры в tensorMap
    std::cout << "[DEBUG] Adding inputs to tensorMap...\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        mlir::Value arg = pImpl->mainFunc.getArgument(i);
        pImpl->tensorMap[inputs[i]] = arg;
        std::cout << "[DEBUG]   " << inputs[i] << " -> argument " << i << "\n";
    }

    std::cout << "[DEBUG] createMainFunction END\n";
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
            outputDimsList.push_back(graph.getTensorDims(outputName));
        }

        if (outputDimsList.empty()) {
            std::cerr << "Error: No output dimensions for node " << nodeId << "\n";
            return false;
        }

        if constexpr (std::is_same_v<T, ConstantNode>) {
            std::cout << "[DEBUG] Emitting ConstantNode\n";
            pImpl->constantEmitter->emitConstant(n.value, n.output_tensors, outputDimsList[0]);
        }
        else if constexpr (std::is_same_v<T, AddNode>) {
            std::cout << "[DEBUG] Emitting AddNode\n";
            if (inputs.size() != 2) {
                std::cerr << "AddNode requires 2 inputs, got " << inputs.size() << "\n";
                return false;
            }
            pImpl->addEmitter->emit(inputs, n.output_tensors, outputDimsList[0]);
        }
        else if constexpr (std::is_same_v<T, MulNode>) {
            std::cout << "[DEBUG] Emitting MulNode\n";
            if (inputs.size() != 2) {
                std::cerr << "MulNode requires 2 inputs, got " << inputs.size() << "\n";
                return false;
            }
            pImpl->mulEmitter->emit(inputs, n.output_tensors, outputDimsList[0]);
        }
        else if constexpr (std::is_same_v<T, ReLUNode>) {
            std::cout << "[DEBUG] Emitting ReluNode\n";
            if (inputs.size() != 1) {
                std::cerr << "ReluNode requires 1 input, got " << inputs.size() << "\n";
                return false;
            }
            pImpl->reluEmitter->emit(inputs, n.output_tensors, outputDimsList[0]);
        }
        else if constexpr (std::is_same_v<T, MatmulNode>) {
            std::cout << "[DEBUG] Emitting MatmulNode\n";
            if (inputs.size() != 2) {
                std::cerr << "MatmulNode requires 2 inputs, got " << inputs.size() << "\n";
                return false;
            }
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
    // ... существующий код до верификации ...

    if (mlir::failed(mlir::verify(*pImpl->module))) {
        std::cerr << "Error: MLIR module verification failed\n";
        return false;
    }

    // ============================================================
    // НОВОЕ: Понижение MLIR до LLVM диалекта
    // ============================================================
    std::cout << "[DEBUG] Running lowering pipeline...\n";
    MLIRPasses passes;
    if (!passes.runLoweringPipeline(*pImpl->module)) {
        std::cerr << "Error: Lowering pipeline failed\n";
        return false;
    }
    std::cout << "[DEBUG] Lowering pipeline completed\n";

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
