# simple_test_model.py
import torch
import torch.nn as nn
import torch.onnx
import onnx
import onnx.checker

class SimpleTestNet(nn.Module):
    """
    Простая модель только с поддерживаемыми операциями:
    - MatMul
    - Add
    - Mul
    - ReLU
    - Constant (через параметры модели)
    """

    def __init__(self):
        super(SimpleTestNet, self).__init__()

        # Константа 1: вес для MatMul (10x5)
        self.weight = nn.Parameter(
            torch.randn(10, 5),
            requires_grad=False
        )

        # Константа 2: bias для Add (1x5)
        self.bias = nn.Parameter(
            torch.ones(1, 5) * 0.5,
            requires_grad=False
        )

        # Константа 3: scale для Mul (1x5)
        self.scale = nn.Parameter(
            torch.ones(1, 5) * 2.0,
            requires_grad=False
        )

        # Константа 4: дополнительный тензор для второго Add
        self.add_const = nn.Parameter(
            torch.ones(1, 5) * 1.0,
            requires_grad=False
        )

    def forward(self, x):
        # x: [batch, 10]

        # 1. MatMul: x * weight -> [batch, 5]
        x = torch.matmul(x, self.weight)

        # 2. Add: добавляем bias
        x = torch.add(x, self.bias)

        # 3. ReLU: активация
        x = torch.relu(x)

        # 4. Mul: умножаем на scale
        x = torch.mul(x, self.scale)

        # 5. Add: добавляем константу
        x = torch.add(x, self.add_const)

        return x


def create_constant_only_model():
    """
    Ещё более простая модель: только Constant + Add
    (для тестирования Constant эмиттера)
    """
    class ConstantOnlyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.const1 = nn.Parameter(torch.ones(5) * 3.0, requires_grad=False)
            self.const2 = nn.Parameter(torch.ones(5) * 2.0, requires_grad=False)

        def forward(self, x):
            # x игнорируется, просто возвращаем сумму констант
            return torch.add(self.const1, self.const2)

    return ConstantOnlyNet()


def create_matmul_only_model():
    """
    Модель только с MatMul (для тестирования MatMul эмиттера)
    """
    class MatMulOnlyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(10, 5), requires_grad=False)

        def forward(self, x):
            return torch.matmul(x, self.weight)

    return MatMulOnlyNet()


def main():
    # Выберите модель для тестирования
    print("Выберите модель:")
    print("1. Полная модель (MatMul + Add + Mul + ReLU)")
    print("2. Только Constant + Add")
    print("3. Только MatMul")

    choice = input("Ваш выбор (1/2/3): ").strip()

    if choice == "2":
        model = create_constant_only_model()
        dummy_input = torch.randn(1, 5)  # не используется, но нужно для экспорта
        print("\n🔧 Создание модели только с Constant + Add...")
    elif choice == "3":
        model = create_matmul_only_model()
        dummy_input = torch.randn(1, 10)
        print("\n🔧 Создание модели только с MatMul...")
    else:
        model = SimpleTestNet()
        dummy_input = torch.randn(1, 10)
        print("\n🔧 Создание полной модели...")

    model.eval()

    output_file = "test_model.onnx"

    print(f"📝 Экспорт в {output_file}...")

    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        export_params=True,        # сохраняем веса как константы
        opset_version=13,
        do_constant_folding=False, # НЕ схлопываем константы (чтобы видеть Constant узлы)
        input_names=['input'],
        output_names=['output'],
        dynamo=False
    )

    # Проверка и вывод структуры
    try:
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("✅ Модель валидна!")

        graph = onnx_model.graph
        print(f"\n📊 Статистика графа:")
        print(f"   Узлов: {len(graph.node)}")
        print(f"   Входы: {[inp.name for inp in graph.input]}")
        print(f"   Выходы: {[out.name for out in graph.output]}")
        print(f"   Initializers (констант): {len(graph.initializer)}")

        print(f"\n🔍 Список операций:")
        for i, node in enumerate(graph.node):
            # Определяем тип входа (initializer или входной тензор)
            inputs_info = []
            for inp_name in node.input:
                is_initializer = any(init.name == inp_name for init in graph.initializer)
                inputs_info.append(f"{inp_name}{'[const]' if is_initializer else ''}")

            print(f"   {i+1}. {node.op_type:8} | inputs: {inputs_info} -> outputs: {node.output}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()
