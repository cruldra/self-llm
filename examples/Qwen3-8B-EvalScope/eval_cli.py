#!/usr/bin/env python3
"""
EvalScope 命令行评测脚本

使用 EvalScope 命令行工具对 Qwen3-8B 模型进行智商情商评测
"""

import os
import subprocess
import sys
import time
import requests
from pathlib import Path


def check_api_server(api_url: str = "http://localhost:8000/health", timeout: int = 5) -> bool:
    """
    检查 API 服务器是否可用
    
    Args:
        api_url: API 健康检查 URL
        timeout: 超时时间
        
    Returns:
        bool: 服务器是否可用
    """
    try:
        response = requests.get(api_url, timeout=timeout)
        return response.status_code == 200
    except:
        return False


def run_evalscope_cli(
    model_name: str = "Qwen3-8B",
    api_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    work_dir: str = "outputs/Qwen3-8B",
    eval_batch_size: int = 16,
    datasets: str = "iquiz"
):
    """
    使用 EvalScope 命令行工具运行评测
    
    Args:
        model_name: 模型名称
        api_url: API 地址
        api_key: API 密钥
        work_dir: 工作目录
        eval_batch_size: 评测批次大小
        datasets: 数据集名称
    """
    print("🧠 使用 EvalScope CLI 进行智商情商评测...")
    print(f"模型: {model_name}")
    print(f"API: {api_url}")
    print(f"数据集: {datasets}")
    print(f"批次大小: {eval_batch_size}")
    print(f"输出目录: {work_dir}")
    print("=" * 50)
    
    # 创建输出目录
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    
    # 构建 evalscope 命令
    cmd = [
        "evalscope", "eval",
        "--model", model_name,
        "--api-url", api_url,
        "--api-key", api_key,
        "--eval-type", "service",
        "--eval-batch-size", str(eval_batch_size),
        "--datasets", datasets,
        "--work-dir", work_dir
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    try:
        # 开始评测
        start_time = time.time()
        print("🚀 开始评测...")
        
        # 执行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ 评测完成！耗时: {duration:.1f} 秒")
        print(f"📊 结果保存在: {work_dir}")
        
        # 显示输出
        if result.stdout:
            print("\n📋 评测输出:")
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 评测失败: {e}")
        if e.stdout:
            print("标准输出:")
            print(e.stdout)
        if e.stderr:
            print("错误输出:")
            print(e.stderr)
        return False
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False


def show_results(work_dir: str = "outputs/Qwen3-8B"):
    """
    显示评测结果
    
    Args:
        work_dir: 工作目录
    """
    print("\n📊 查看评测结果...")
    
    work_path = Path(work_dir)
    if not work_path.exists():
        print(f"❌ 结果目录不存在: {work_dir}")
        return
    
    # 查找最新的评测结果目录
    result_dirs = [d for d in work_path.iterdir() if d.is_dir()]
    if not result_dirs:
        print(f"❌ 未找到评测结果")
        return
    
    latest_dir = max(result_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📁 最新结果目录: {latest_dir}")
    
    # 显示目录结构
    print("\n📂 结果文件:")
    for item in latest_dir.rglob("*"):
        if item.is_file():
            relative_path = item.relative_to(latest_dir)
            size = item.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  📄 {relative_path} ({size_str})")
    
    # 查找并显示摘要文件
    summary_files = list(latest_dir.rglob("*summary*"))
    if summary_files:
        print(f"\n📋 评测摘要文件:")
        for file in summary_files:
            print(f"  📊 {file}")
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) < 1000:  # 只显示小文件内容
                        print(f"内容:\n{content}")
                    else:
                        print(f"文件较大，请直接查看: {file}")
            except Exception as e:
                print(f"读取文件失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EvalScope CLI 智商情商评测")
    parser.add_argument("--model-name", default="Qwen3-8B", help="模型名称")
    parser.add_argument("--api-url", default="http://localhost:8000/v1", help="API 地址")
    parser.add_argument("--api-key", default="EMPTY", help="API 密钥")
    parser.add_argument("--work-dir", default="outputs/Qwen3-8B", help="工作目录")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="评测批次大小")
    parser.add_argument("--datasets", default="iquiz", help="数据集名称")
    parser.add_argument("--skip-check", action="store_true", help="跳过 API 服务检查")
    
    args = parser.parse_args()
    
    # 检查 evalscope 命令是否可用
    try:
        subprocess.run(["evalscope", "--help"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ evalscope 命令不可用")
        print("请安装 EvalScope: pip install evalscope")
        sys.exit(1)
    
    # 检查 API 服务
    if not args.skip_check:
        health_url = args.api_url.replace("/v1", "/health")
        if not check_api_server(health_url):
            print(f"❌ API 服务不可用: {health_url}")
            print("请先启动 vLLM 服务: uv run python start_vllm_server.py")
            sys.exit(1)
        print(f"✅ API 服务正常: {health_url}")
    
    # 运行评测
    success = run_evalscope_cli(
        model_name=args.model_name,
        api_url=args.api_url,
        api_key=args.api_key,
        work_dir=args.work_dir,
        eval_batch_size=args.eval_batch_size,
        datasets=args.datasets
    )
    
    if success:
        # 显示结果
        show_results(args.work_dir)
        
        print("\n🎉 评测完成！")
        print(f"📊 详细结果请查看: {args.work_dir}")
        print("\n💡 提示:")
        print("- IQuiz 数据集包含智商和情商测试题")
        print("- 可以通过调整批次大小优化评测速度")
        print("- 评测结果包含详细的答题分析")
    else:
        print("\n❌ 评测失败！")
        print("请检查:")
        print("1. EvalScope 是否正确安装")
        print("2. vLLM 服务是否正常运行")
        print("3. 模型是否正确加载")


if __name__ == "__main__":
    main()
