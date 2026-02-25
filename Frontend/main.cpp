#include <iostream>
#include <fstream>

#include "onnx/onnx.pb.h"

int main() {
    onnx::ModelProto model;

    std::ifstream model_file("data/yolov8m.onnx");

    model.ParseFromIstream(&model_file);

    std::cout << "Heh\n";
    return 0;
}
