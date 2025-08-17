"""
Qwen3-8B 模型下载脚本

使用 modelscope 下载 Qwen3-8B 模型到本地
"""

import os
from pathlib import Path
from modelscope import snapshot_download


def download_qwen3_model(
    model_name: str = "Qwen/Qwen3-8B",
    cache_dir: str = "./models",
    revision: str = "master"
) -> str:
    """
    下载 Qwen3-8B 模型
    
    Args:
        model_name: 模型名称，默认为 "Qwen/Qwen3-8B"
        cache_dir: 模型缓存目录，默认为 "./models"
        revision: 模型版本，默认为 "master"
    
    Returns:
        str: 模型下载路径
    """
    print(f"开始下载模型: {model_name}")
    print(f"下载目录: {cache_dir}")
    print(f"模型版本: {revision}")
    
    # 确保下载目录存在
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # 下载模型
        model_dir = snapshot_download(
            model_name,
            cache_dir=cache_dir,
            revision=revision
        )
        
        print(f"模型下载完成！")
        print(f"模型路径: {model_dir}")
        
        return model_dir
        
    except Exception as e:
        print(f"模型下载失败: {e}")
        raise


def check_model_files(model_dir: str) -> bool:
    """
    检查模型文件是否完整
    
    Args:
        model_dir: 模型目录路径
    
    Returns:
        bool: 文件是否完整
    """
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt"
    ]
    
    model_path = Path(model_dir)
    missing_files = []
    
    for file_name in required_files:
        if not (model_path / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"缺少以下文件: {missing_files}")
        return False
    
    print("模型文件检查完成，所有必需文件都存在")
    return True


def main():
    """主函数"""
    # 配置参数
    model_name = "Qwen/Qwen3-8B"
    cache_dir = "./models"
    
    # 如果是在 AutoDL 环境，使用 /root/autodl-tmp
    if os.path.exists("/root/autodl-tmp"):
        cache_dir = "/root/autodl-tmp"
        print("检测到 AutoDL 环境，使用 /root/autodl-tmp 作为下载目录")
    
    try:
        # 下载模型
        model_dir = download_qwen3_model(
            model_name=model_name,
            cache_dir=cache_dir
        )
        
        # 检查文件完整性
        if check_model_files(model_dir):
            print("\n✅ 模型下载和验证完成！")
            print(f"📁 模型路径: {model_dir}")
            print("\n🚀 现在可以运行推理脚本了:")
            print("   uv run python vllm_thinking_mode.py")
            print("   uv run python vllm_non_thinking_mode.py")
        else:
            print("\n❌ 模型文件不完整，请重新下载")
            
    except Exception as e:
        print(f"\n❌ 下载过程中出现错误: {e}")
        print("请检查网络连接和磁盘空间")


if __name__ == "__main__":
    main()
