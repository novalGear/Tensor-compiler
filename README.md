# Tensor-compiler

1) нейронку на питоне накидать и сохранить в onnx
2) Portobuf разобраться, получить представление нейронки
3) написать свой граф, с поддержкой операций
4) конвертация из обычного представления в наше
5) graphviz визуализация графа

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

## TODO:
- наследование -> аргумент в виде std::variant
- валидация построения графа (строить граф по Protobuf графу и сравнивать)?
- нормальный вывод в dot с паарметрами узла
- доки на код
- побольше тестовых нейронок
