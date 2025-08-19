"""
Qwen3-8B vLLM API 服务器启动脚本

使用 vLLM 启动兼容 OpenAI API 的服务器
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def get_model_path() -> str:
    """获取模型路径"""
    # 优先检查 AutoDL 环境
    autodl_path = "/root/autodl-tmp/Qwen/Qwen3-8B"
    if os.path.exists(autodl_path):
        return autodl_path
    
    # 检查本地 models 目录
    local_path = "./models/Qwen/Qwen3-8B"
    if os.path.exists(local_path):
        return local_path
    
    # 如果都不存在，返回默认路径并提示用户
    print("⚠️  未找到模型文件，请先运行 model_download.py 下载模型")
    return local_path


def check_model_exists(model_path: str) -> bool:
    """检查模型是否存在"""
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        print("请先运行以下命令下载模型:")
        print("   uv run python model_download.py")
        return False
    
    # 检查关键文件
    required_files = ["config.json", "tokenizer.json"]
    for file_name in required_files:
        if not os.path.exists(os.path.join(model_path, file_name)):
            print(f"❌ 缺少模型文件: {file_name}")
            return False
    
    return True


def start_vllm_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    served_model_name: str = "Qwen3-8B",
    max_model_len: int = 8192,
    enable_reasoning: bool = True,
    reasoning_parser: str = "qwen3",
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1
):
    """
    启动 vLLM API 服务器
    
    Args:
        model_path: 模型路径
        host: 服务器主机地址
        port: 服务器端口
        served_model_name: 服务模型名称
        max_model_len: 模型最大长度
        enable_reasoning: 是否启用推理模式
        reasoning_parser: 推理解析器
        gpu_memory_utilization: GPU 内存利用率
        tensor_parallel_size: 张量并行大小
    """
    # 设置环境变量
    env = os.environ.copy()
    env['VLLM_USE_MODELSCOPE'] = 'true'
    
    # 构建命令
    cmd = [
        "vllm", "serve", model_path,
        "--host", host,
        "--port", str(port),
        "--served-model-name", served_model_name,
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--trust-remote-code"
    ]
    
    # 添加推理模式参数
    if enable_reasoning:
        cmd.extend(["--reasoning-parser", reasoning_parser])
    
    print("🚀 启动 vLLM API 服务器")
    print("=" * 50)
    print(f"📁 模型路径: {model_path}")
    print(f"🌐 服务地址: http://{host}:{port}")
    print(f"🏷️  模型名称: {served_model_name}")
    print(f"📏 最大长度: {max_model_len}")
    print(f"🧠 推理模式: {'启用' if enable_reasoning else '禁用'}")
    print(f"🎯 GPU 利用率: {gpu_memory_utilization}")
    print(f"⚡ 并行大小: {tensor_parallel_size}")
    print("=" * 50)
    
    print("执行命令:")
    print(" ".join(cmd))
    print("\n⏳ 正在启动服务器...")
    print("💡 提示: 服务器启动后，可以通过以下方式测试:")
    print(f"   curl http://localhost:{port}/v1/models")
    print(f"   uv run python test_openai_chat_completions.py")
    print("\n按 Ctrl+C 停止服务器")
    print("-" * 50)
    
    try:
        # 启动服务器
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 服务器启动失败: {e}")
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动 Qwen3-8B vLLM API 服务器")
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="模型路径 (默认自动检测)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器主机地址 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口 (默认: 8000)"
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default="Qwen3-8B",
        help="服务模型名称 (默认: Qwen3-8B)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="模型最大长度 (默认: 8192)"
    )
    parser.add_argument(
        "--disable-reasoning",
        action="store_true",
        help="禁用推理模式"
    )
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default="qwen3",
        help="推理解析器 (默认: qwen3)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU 内存利用率 (默认: 0.9)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="张量并行大小 (默认: 1)"
    )
    
    args = parser.parse_args()
    
    # 获取模型路径
    model_path = args.model_path or get_model_path()
    
    # 检查模型是否存在
    if not check_model_exists(model_path):
        sys.exit(1)
    
    # 启动服务器
    start_vllm_server(
        model_path=model_path,
        host=args.host,
        port=args.port,
        served_model_name=args.served_model_name,
        max_model_len=args.max_model_len,
        enable_reasoning=not args.disable_reasoning,
        reasoning_parser=args.reasoning_parser,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size
    )


if __name__ == "__main__":
    main()
