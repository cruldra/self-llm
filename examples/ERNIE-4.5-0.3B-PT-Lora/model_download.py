"""
ERNIE-4.5-0.3B-PT 模型下载脚本

使用 modelscope 从魔搭平台下载 ERNIE-4.5-0.3B-PT 模型
"""

import os

from config import ModelDownloadConfig, Paths
from modelscope import snapshot_download


def download_model():
    """下载 ERNIE-4.5-0.3B-PT 模型"""

    # 确保目录存在
    os.makedirs(Paths.CACHE_DIR, exist_ok=True)

    print(f"开始下载 ERNIE-4.5-0.3B-PT 模型到: {Paths.CACHE_DIR}")
    print("模型大小约 57GB，下载时间较长，请耐心等待...")

    try:
        model_dir = snapshot_download(
            ModelDownloadConfig.MODEL_NAME,
            cache_dir=Paths.CACHE_DIR,
            revision=ModelDownloadConfig.REVISION
        )
        print(f"模型下载完成！模型路径: {model_dir}")
        return model_dir
    except Exception as e:
        print(f"模型下载失败: {e}")
        return None

if __name__ == "__main__":
    download_model()
