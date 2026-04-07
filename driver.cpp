// driver.cpp
#include "graph.hpp"
#include "MiddleEnd/MLIR/MLIRGenerator.hpp"
#include <iostream>
#include <string>
#include <cstring>
#include <memory>

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " <input.onnx> [options]\n\n"
              << "Options:\n"
              << "  --emit-mlir          Print generated MLIR to stdout\n"
              << "  --save-mlir <file>   Save MLIR to file\n"
              << "  --output <file>      Output file prefix (default: output)\n"
              << "  -h, --help           Show this help\n\n"
              << "Examples:\n"
              << "  " << progName << " model.onnx --emit-mlir\n"
              << "  " << progName << " model.onnx --save-mlir model.mlir\n";
}

struct Options {
    std::string inputFile;
    std::string outputPrefix = "output";
    std::string saveMlirFile = "";
    bool emitMlir = false;
    bool help = false;
    bool verbose = false;
};

Options parseArgs(int argc, char** argv) {
    Options opts;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--emit-mlir") == 0) {
            opts.emitMlir = true;
        }
        else if (strcmp(argv[i], "--save-mlir") == 0 && i + 1 < argc) {
            opts.saveMlirFile = argv[++i];
        }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            opts.outputPrefix = argv[++i];
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            opts.verbose = true;
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            opts.help = true;
        }
        else if (argv[i][0] != '-') {
            opts.inputFile = argv[i];
        }
        else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            opts.help = true;
        }
    }

    return opts;
}

int main(int argc, char** argv) {
    auto opts = parseArgs(argc, argv);

    if (opts.help || opts.inputFile.empty()) {
        printUsage(argv[0]);
        return opts.help ? 0 : 1;
    }

    std::cout << "Loading ONNX model: " << opts.inputFile << "\n";

    // Загрузка графа из ONNX
    auto graph = tcc::ComputeGraph::load_from_onnx(opts.inputFile);
    if (!graph) {
        std::cerr << "Error: Failed to load ONNX model\n";
        return 1;
    }

    // Диагностика графа (если включен verbose режим)
    if (opts.verbose) {
        graph->print_graph_info();
    }

    // Настройка MLIR генератора
    tcc::mlir_gen::MLIRGenerator::Config cfg;
    cfg.printMLIR = opts.emitMlir;
    cfg.outputFile = opts.saveMlirFile;

    // Генерация MLIR
    tcc::mlir_gen::MLIRGenerator generator(cfg);
    if (!generator.generate(*graph)) {
        std::cerr << "Error: MLIR generation failed\n";
        return 1;
    }

    std::cout << "MLIR generation successful!\n";

    return 0;
}
