# Tensor-compiler

## Зависимости

Для сборки проекта требуется компилятор C++17, система сборки CMake и библиотеки Protobuf.

Установите необходимые пакеты в WSL/Linux:

```bash
sudo apt update
sudo apt install -y build-essential cmake git protobuf-compiler libprotobuf-dev pkg-config
```

## Сборка проекта

```bash
# сборка
mkdir -p build && cd build && cmake ..
# компиляция
make -j10
```

## Запуск

```bash
./build/Frontend/frontend ./data/test_model.onnx
```

или указывая путь к другому .onnx в качестве аргумента

## Frontend: структура проекта

Типы узлов описываются в 'Frontend/ops_def.json', в следующем формате:

```json
  {
    "op_type":  "Conv",
    "cpp_name": "ConvNode",
    "attributes": [
      { "name": "strides",      "type": "std::vector<int64_t>", "default": "{1}"    },
      { "name": "pads",         "type": "std::vector<int64_t>", "default": "{0}"    },
      { "name": "dilations",    "type": "std::vector<int64_t>", "default": "{1}"    },
      { "name": "group",        "type": "int64_t",              "default": "1"      }
    ]
  },
```

Скрипт `Frontend/generate_nodes.rb` генерирует по данному описанию:
- `Frontend/Graph/graph_gen.hpp`: описание структур узлов
- `Frontend/Graph/graph_gen_parser.inl`: функции конвертации для каждого типа узла
- `Frontend/Graph/graph_gen_utils.inl`: функции для вывода в dot для каждого типа узла

Описание графа и тензоров находятся в `graph.hpp`, реализации методов в `convertion.cpp`
