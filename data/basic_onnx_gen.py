import torch
import torch.nn as nn
import torch.onnx
import onnx
import onnx.checker

class ComprehensiveNet(nn.Module):
    def __init__(self):
        super(ComprehensiveNet, self).__init__()

        # 1. Conv2d: Будет иметь атрибуты stride, padding, dilation
        # Вход: [1, 3, 32, 32] -> Выход: [1, 8, 14, 14]
        # Формула размера: (32 - 3 + 2*1) / 2 + 1 = 15? Нет, давайте подберем точно.
        # Kernel=3, Stride=2, Padding=1 -> (32 - 3 + 2)/2 + 1 = 16. Ок.
        self.conv_layer = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3,
                                    stride=2, padding=1, bias=True)

        # 2. Relu: Активация после свертки
        self.relu_layer = nn.ReLU()

        # Подготовим веса для остальных операций, чтобы они были константами в графе
        # 3. Gemm: Обычно это Linear слой.
        # Вход после Flatten: 8 * 16 * 16 = 2048. Сделаем выход 64.
        self.linear_layer = nn.Linear(2048, 64)

        # Для демонстрации чистого MatMul, Add, Mul нам нужно немного "хакнуть" forward,
        # так как стандартные слои PyTorch часто фузят операции.
        # Мы создадим буферные тензоры-константы прямо в модуле.

        # Матрица для MatMul (64 x 10)
        self.matmul_weight = nn.Parameter(torch.randn(64, 10), requires_grad=False)

        # Вектор для Add (размер 10)
        self.add_bias = nn.Parameter(torch.ones(10), requires_grad=False)

        # Скаляр/Вектор для Mul (размер 10)
        self.mul_scale = nn.Parameter(torch.ones(10) * 2.0, requires_grad=False)

    def forward(self, x):
        # --- Узел 1: Conv ---
        # Атрибуты: stride=[2,2], pads=[1,1,1,1], dilations=[1,1]
        x = self.conv_layer(x)

        # --- Узел 2: Relu ---
        x = self.relu_layer(x)

        # Подготовка к линейным операциям: Flatten
        # Размер становится [Batch, 8*16*16] = [1, 2048]
        x = torch.flatten(x, 1)

        # --- Узел 3: Gemm ---
        # PyTorch Linear реализует: y = x * A^T + b.
        # В ONNX это часто раскладывается на Gemm или MatMul+Add.
        # Экспортер должен создать узел Gemm с атрибутами transB=1.
        x = self.linear_layer(x)

        # --- Узел 4: MatMul ---
        # Умножаем результат (1x64) на вес (64x10) -> (1x10)
        # Это создаст чистый узел MatMul в графе
        x = torch.matmul(x, self.matmul_weight)

        # --- Узел 5: Add ---
        # Прибавляем константу (1x10) + (1x10)
        # Создаст узел Add
        x = torch.add(x, self.add_bias)

        # --- Узел 6: Mul ---
        # Умножаем на скаляр/вектор
        # Создаст узел Mul
        x = torch.mul(x, self.mul_scale)

        return x

def main():
    # Инициализация модели
    model = ComprehensiveNet()
    model.eval() # Режим инференса важен для экспорта

    # Создание фиктивного входа (Batch=1, Channels=3, H=32, W=32)
    dummy_input = torch.randn(1, 3, 32, 32)

    # Имя выходного файла
    output_file = "test_model.onnx"

    print(f"Генерация модели {output_file}...")

    # Экспорт в ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        export_params=True,        # Сохранять веса
        opset_version=13,          # Современная версия опсета
        do_constant_folding=False, # Не схлопывать константы, чтобы видеть все узлы явно
        input_names=['input_tensor'],
        output_names=['output_tensor'],
        dynamo=False
    )

    # Проверка валидности созданного файла
    try:
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("✅ Модель успешно создана и проверена!")

        # Краткий вывод структуры для проверки
        graph = onnx_model.graph
        print(f"\n📊 Статистика графа:")
        print(f"   Количество узлов: {len(graph.node)}")
        print(f"   Входы: {[inp.name for inp in graph.input]}")
        print(f"   Выходы: {[out.name for out in graph.output]}")

        print(f"\n🔍 Список операций в графе:")
        for i, node in enumerate(graph.node):
            attrs = ", ".join([f"{a.name}={a.i if a.type==2 else a.f}" for a in node.attribute])
            print(f"   {i+1}. {node.op_type:6} | Out: {node.output[0]:15} | Attrs: [{attrs}]")

    except Exception as e:
        print(f"❌ Ошибка при проверке модели: {e}")

if __name__ == "__main__":
    main()
