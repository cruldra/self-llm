# Gemma-3-4b-it FastAPI 部署示例

这是一个基于 FastAPI 的 Gemma-3-4b-it 模型部署示例项目，使用 uv 进行依赖管理。

## 环境要求

- Python 3.12+
- CUDA 12.1+ (用于 GPU 推理)
- uv (Python 包管理器)

## 安装依赖

本项目使用 uv 管理依赖，确保您已安装 uv：

```bash
# 安装 uv (如果尚未安装)
pip install uv
```

然后安装项目依赖：

```bash
# 激活虚拟环境并安装依赖
uv sync
```

## 模型下载

运行以下命令下载 Gemma-3-4b-it 模型：

```bash
uv run python model_download.py
```

注意：请根据您的实际情况修改 `model_download.py` 中的 `cache_dir` 路径。

## 启动服务

```bash
uv run python api.py
```

服务将在 `http://0.0.0.0:6006` 启动。

## API 使用

### 纯文本对话

```bash
curl -X POST "http://127.0.0.1:6006/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{
       "messages": [
         {
           "role": "user",
           "content": [
             {"type": "text", "text": "帮我写一份科幻小说的大纲！"}
           ]
         }
       ],
       "max_new_tokens": 4096
     }'
```

### 图片+文本对话

```bash
curl -X POST "http://127.0.0.1:6006/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{
       "messages": [
         {
           "role": "user",
           "content": [
             {"type": "image", "image": "http://example.com/image.jpg"},
             {"type": "text", "text": "请描述图片中的详情信息"}
           ]
         }
       ],
       "max_new_tokens": 4096
     }'
```

## 测试客户端

运行测试客户端：

```bash
uv run python test_client.py
```

## 项目结构

```
gemma-3-4b-it-FastApi/
├── api.py              # FastAPI 服务主文件
├── model_download.py   # 模型下载脚本
├── test_client.py      # 测试客户端
├── pyproject.toml      # uv 项目配置文件
├── README.md           # 项目说明
└── models/             # 模型存储目录（运行后生成）
```

## 支持的消息类型

1. **纯文本对话**：只包含文本内容
2. **图片+文本对话**：包含图片URL和文本描述
3. **Base64图片+文本对话**：包含base64编码的图片和文本

## 注意事项

- 确保有足够的 GPU 内存来加载模型
- 首次运行时会下载模型文件，请确保网络连接稳定
- 模型路径可以根据实际情况进行调整