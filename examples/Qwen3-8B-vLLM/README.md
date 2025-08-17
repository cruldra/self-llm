# Qwen3-8B vLLM 部署调用示例

本示例项目演示如何使用 vLLM 部署和调用 Qwen3-8B 模型，包含完整的代码示例和使用说明。

## 项目结构

```
Qwen3-8B-vLLM/
├── README.md                           # 项目说明文档
├── pyproject.toml                      # Python 项目配置文件
├── requirements.txt                    # 依赖列表
├── model_download.py                   # 模型下载脚本
├── vllm_basic_inference.py            # 基础推理示例
├── vllm_thinking_mode.py              # 思考模式示例
├── vllm_non_thinking_mode.py          # 非思考模式示例
├── start_api_server.py                # 启动 API 服务器脚本
├── test_openai_completions.py         # OpenAI Completions API 测试
├── test_openai_chat_completions.py    # OpenAI Chat Completions API 测试
├── scripts/
│   ├── start_server.sh                # 启动服务器脚本
│   └── test_api.sh                    # API 测试脚本
└── examples/
    ├── basic_chat.py                  # 基础对话示例
    ├── batch_inference.py             # 批量推理示例
    └── streaming_chat.py              # 流式对话示例
```

## 环境要求

- Python 3.12+
- CUDA 12.4+
- PyTorch 2.5.1+
- 至少 16GB GPU 显存

## 快速开始

### 1. 安装依赖

```bash
# 使用 uv 安装依赖
uv sync

# 或使用 pip 安装
pip install -r requirements.txt
```

### 2. 下载模型

```bash
uv run python model_download.py
```

### 3. 基础推理测试

```bash
# 思考模式推理
uv run python vllm_thinking_mode.py

# 非思考模式推理
uv run python vllm_non_thinking_mode.py
```

### 4. 启动 API 服务器

```bash
# 使用 Python 脚本启动
uv run python start_api_server.py

# 或使用 shell 脚本启动
bash scripts/start_server.sh
```

### 5. 测试 API 接口

```bash
# 测试 OpenAI API
uv run python test_openai_completions.py
uv run python test_openai_chat_completions.py

# 或使用 curl 测试
bash scripts/test_api.sh
```

## 功能特性

### vLLM 核心特性
- **高效内存管理**：通过 PagedAttention 算法优化 KV 缓存
- **高吞吐量**：支持异步处理和连续批处理
- **易用性**：与 HuggingFace 模型无缝集成
- **分布式推理**：支持多 GPU 环境
- **OpenAI 兼容**：完全兼容 OpenAI API 协议

### Qwen3-8B 特性
- **思考模式**：类似 QwQ-32B 的推理能力
- **多语言支持**：中英文对话能力
- **代码生成**：支持多种编程语言
- **数学推理**：强化的数学计算能力

## 使用示例

### 基础推理
```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(model="path/to/Qwen3-8B")

# 生成文本
outputs = llm.generate(prompts, sampling_params)
```

### API 调用
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-xxx"
)

response = client.chat.completions.create(
    model="Qwen3-8B",
    messages=[{"role": "user", "content": "你好"}]
)
```

## 配置说明

### 思考模式参数
- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`
- `min_p=0`

### 非思考模式参数
- `temperature=0.7`
- `top_p=0.8`
- `top_k=20`
- `min_p=0`

## 故障排除

### 常见问题
1. **显存不足**：调整 `max_model_len` 参数
2. **模型加载失败**：检查模型路径和权限
3. **API 连接失败**：确认服务器已启动且端口未被占用

### 性能优化
1. 使用多 GPU 并行推理
2. 调整批处理大小
3. 优化采样参数

## 参考资料

- [vLLM 官方文档](https://docs.vllm.ai/)
- [Qwen3 模型介绍](https://github.com/QwenLM/Qwen3)
- [OpenAI API 文档](https://platform.openai.com/docs/api-reference)

## 许可证

本项目遵循 Apache 2.0 许可证。
