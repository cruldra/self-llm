from modelscope import snapshot_download

# 下载 Gemma-3-4b-it 模型
# 注意：请根据您的实际情况修改 cache_dir 路径
model_dir = snapshot_download('LLM-Research/gemma-3-4b-it', cache_dir='./models', revision='master')
print(f"模型已下载到: {model_dir}")
