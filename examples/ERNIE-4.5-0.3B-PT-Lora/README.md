# ERNIE-4.5-0.3B-PT LoRA 微调示例

本项目演示如何使用 LoRA 技术对 ERNIE-4.5-0.3B-PT 模型进行微调，构建一个能够模拟甄嬛对话风格的个性化 LLM，并使用 SwanLab 进行训练过程可视化。

## 项目特点

- 🚀 使用 uv 进行快速依赖管理
- 📊 集成 SwanLab 进行训练可视化
- 🎯 基于甄嬛数据集的个性化对话微调
- 💡 高效的 LoRA 微调技术
- 🔧 完整的训练和推理流程

## 环境要求

- Python 3.12+
- CUDA 12.4+
- PyTorch 2.5.1+
- 显存需求：约 24GB

## 快速开始

### 1. 环境配置

使用 uv 创建虚拟环境并安装依赖：

```bash
# 创建虚拟环境并安装依赖
uv sync

# 安装带有cuda支持的torch
uv run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 2. 模型下载

运行模型下载脚本：

```bash
uv run python model_download.py
```

### 3. 开始训练

```bash
uv run python train.py
```

### 4. 推理测试

```bash
uv run python inference.py
```

### 💡 关于 uv run 命令

使用 `uv run` 命令的优势：
- 🚀 自动激活虚拟环境
- ✅ 确保使用正确的 Python 解释器和依赖
- 🔧 无需手动激活/停用虚拟环境
- 📦 更简洁的工作流程

如果您更喜欢传统方式，也可以手动激活虚拟环境：
```bash
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 然后直接运行 Python 脚本
python model_download.py
python train.py
python inference.py
```

## 项目结构

```
ERNIE-4.5-0.3B-PT-Lora/
├── README.md              # 项目说明
├── pyproject.toml          # 项目配置和依赖
├── model_download.py       # 模型下载脚本
├── train.py               # 训练脚本
├── inference.py           # 推理脚本
├── models/                # 模型存储目录
└── output/                # 训练输出目录
```

## 数据集

本项目使用 [Chat-甄嬛](https://github.com/KMnO4-zx/huanhuan-chat) 项目中的甄嬛数据集，数据集位于 `../../dataset/huanhuan.json`。

## 训练配置

- **LoRA 秩 (r)**: 8
- **LoRA Alpha**: 32
- **Dropout**: 0.1
- **学习率**: 1e-4
- **训练轮次**: 3
- **批量大小**: 4
- **梯度累积步数**: 4

## SwanLab 可视化

项目集成了 SwanLab 进行训练过程可视化，可以实时监控：
- 训练损失
- 学习率变化
- 梯度变化
- 模型参数

## 注意事项

1. 请确保有足够的显存（约24GB）
2. 修改脚本中的模型路径为你的实际路径
3. 首次运行需要下载模型，请确保网络连接稳定

## 参考资料

- [ERNIE-4.5-0.3B-PT 模型](https://www.modelscope.cn/models/PaddlePaddle/ERNIE-4.5-0.3B-PT)
- [LoRA 原理详解](https://zhuanlan.zhihu.com/p/650197598)
- [SwanLab 官网](https://swanlab.cn/)
