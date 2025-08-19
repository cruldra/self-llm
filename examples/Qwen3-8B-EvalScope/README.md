# Qwen3-8B EvalScope 智商情商评测示例

本示例项目演示如何使用 EvalScope 框架对 Qwen3-8B 模型进行智商情商评测。

## 项目简介

EvalScope 是魔搭社区官方推出的模型评测与性能基准测试框架，内置多个常用测试基准和评测指标。本项目专门针对 Qwen3-8B 模型的智商情商评测，使用 IQuiz 数据集进行测试。

## 环境要求

- Python >= 3.8
- CUDA >= 12.0 (推荐)
- 内存 >= 16GB
- 显存 >= 16GB

## 快速开始

### 1. 安装依赖

```bash
# 使用 uv 安装依赖
uv sync

# 或者使用 pip 安装
pip install -e .
```

### 2. 下载模型

```bash
uv run python model_download.py
```

### 3. 启动 vLLM 服务

```bash
# 方式一：使用脚本启动
bash scripts/start_vllm_server.sh

# 方式二：直接命令启动
uv run python start_vllm_server.py
```

### 4. 运行评测

```bash
# 方式一：使用命令行评测
uv run python eval_cli.py

# 方式二：使用 Python API 评测
uv run python eval_api.py

# 方式三：使用 EvalScope 命令行
evalscope eval \
  --model Qwen3-8B \
  --api-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --eval-type service \
  --eval-batch-size 16 \
  --datasets iquiz \
  --work-dir outputs/Qwen3-8B
```

## 项目结构

```
Qwen3-8B-EvalScope/
├── README.md                    # 项目说明
├── pyproject.toml              # 项目配置
├── model_download.py           # 模型下载脚本
├── start_vllm_server.py        # vLLM 服务启动脚本
├── eval_api.py                 # Python API 评测脚本
├── eval_cli.py                 # 命令行评测脚本
├── config/                     # 配置文件目录
│   ├── eval_config.py         # 评测配置
│   └── model_config.py        # 模型配置
├── scripts/                    # 脚本目录
│   ├── start_vllm_server.sh   # vLLM 服务启动脚本
│   └── run_evaluation.sh      # 评测运行脚本
├── examples/                   # 示例代码
│   ├── basic_eval.py          # 基础评测示例
│   ├── custom_eval.py         # 自定义评测示例
│   └── batch_eval.py          # 批量评测示例
├── outputs/                    # 评测结果输出目录
└── models/                     # 模型存储目录
    └── Qwen/
        └── Qwen3-8B/
```

## 评测数据集

本项目使用 IQuiz 数据集，包含：
- 40 道 IQ 测试题
- 80 道 EQ 测试题

数据集包含一些经典问题，如：
- 数字比较问题
- 字符统计问题
- 情商理解问题

## 评测结果

评测完成后，结果将保存在 `outputs/` 目录下，包含：
- 详细的评测报告
- 各项指标得分
- 错误分析
- 模型回答详情

## 注意事项

1. 确保有足够的显存运行 Qwen3-8B 模型
2. 评测过程中 temperature 参数会影响结果稳定性
3. 建议在评测前先测试 vLLM 服务是否正常运行
4. 评测时间约 3-5 分钟，具体取决于硬件配置

## 参考资料

- [EvalScope 官方文档](https://evalscope.readthedocs.io/zh-cn/latest/)
- [Qwen3 评测最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/qwen3.html)
- [IQuiz 数据集](https://modelscope.cn/datasets/AI-ModelScope/IQuiz)
