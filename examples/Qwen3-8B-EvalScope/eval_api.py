#!/usr/bin/env python3
"""
EvalScope Python API 评测脚本

使用 EvalScope 框架对 Qwen3-8B 模型进行智商情商评测
"""

import os
import time
import requests
from pathlib import Path
from evalscope.run import run_task
from evalscope.config import TaskConfig


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


def run_iquiz_evaluation(
    model_name: str = "Qwen3-8B",
    api_url: str = "http://localhost:8000/v1/chat/completions",
    api_key: str = "EMPTY",
    work_dir: str = "outputs/Qwen3-8B",
    temperature: float = 0.7,
    max_tokens: int = 4096
):
    """
    运行 IQuiz 智商情商评测
    
    Args:
        model_name: 模型名称
        api_url: API 地址
        api_key: API 密钥
        work_dir: 工作目录
        temperature: 温度参数
        max_tokens: 最大 token 数
    """
    print("🧠 开始 Qwen3-8B 智商情商评测...")
    print(f"模型: {model_name}")
    print(f"API: {api_url}")
    print(f"输出目录: {work_dir}")
    print("=" * 50)
    
    # 创建输出目录
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    
    # 配置评测任务
    task_cfg = TaskConfig(
        model=model_name,
        api_url=api_url,
        api_key=api_key,
        eval_type='service',
        datasets=['iquiz'],  # IQuiz 智商情商测试数据集
        generation_config={
            'max_tokens': max_tokens,
            'max_new_tokens': max_tokens,
            'temperature': temperature,
        },
        work_dir=work_dir,
    )
    
    print("📋 评测配置:")
    print(f"  数据集: {task_cfg.datasets}")
    print(f"  温度参数: {temperature}")
    print(f"  最大 tokens: {max_tokens}")
    print(f"  评测类型: {task_cfg.eval_type}")
    print()
    
    try:
        # 开始评测
        start_time = time.time()
        print("🚀 开始评测...")
        
        # 执行评测任务
        result = run_task(task_cfg=task_cfg)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ 评测完成！耗时: {duration:.1f} 秒")
        print(f"📊 结果保存在: {work_dir}")
        
        # 显示评测结果
        if result:
            print("\n📈 评测结果:")
            for key, value in result.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
        
        return result
        
    except Exception as e:
        print(f"❌ 评测失败: {e}")
        return None


def analyze_results(work_dir: str = "outputs/Qwen3-8B"):
    """
    分析评测结果
    
    Args:
        work_dir: 工作目录
    """
    print("\n📊 分析评测结果...")
    
    # 查找结果文件
    work_path = Path(work_dir)
    if not work_path.exists():
        print(f"❌ 结果目录不存在: {work_dir}")
        return
    
    # 查找最新的评测结果
    result_dirs = [d for d in work_path.iterdir() if d.is_dir()]
    if not result_dirs:
        print(f"❌ 未找到评测结果")
        return
    
    latest_dir = max(result_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📁 最新结果目录: {latest_dir}")
    
    # 查找结果文件
    result_files = list(latest_dir.rglob("*.json"))
    if result_files:
        print(f"📄 结果文件数量: {len(result_files)}")
        for file in result_files[:5]:  # 显示前5个文件
            print(f"  - {file.name}")
    
    # 查找评测报告
    report_files = list(latest_dir.rglob("*summary*"))
    if report_files:
        print(f"📋 评测报告:")
        for file in report_files:
            print(f"  - {file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3-8B EvalScope 智商情商评测")
    parser.add_argument("--model-name", default="Qwen3-8B", help="模型名称")
    parser.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions", help="API 地址")
    parser.add_argument("--api-key", default="EMPTY", help="API 密钥")
    parser.add_argument("--work-dir", default="outputs/Qwen3-8B", help="工作目录")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
    parser.add_argument("--max-tokens", type=int, default=4096, help="最大 tokens")
    parser.add_argument("--skip-check", action="store_true", help="跳过 API 服务检查")
    
    args = parser.parse_args()
    
    # 检查 API 服务
    if not args.skip_check:
        health_url = args.api_url.replace("/v1/chat/completions", "/health")
        if not check_api_server(health_url):
            print(f"❌ API 服务不可用: {health_url}")
            print("请先启动 vLLM 服务: uv run python start_vllm_server.py")
            return
        print(f"✅ API 服务正常: {health_url}")
    
    # 运行评测
    result = run_iquiz_evaluation(
        model_name=args.model_name,
        api_url=args.api_url,
        api_key=args.api_key,
        work_dir=args.work_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    if result:
        # 分析结果
        analyze_results(args.work_dir)
        
        print("\n🎉 评测完成！")
        print(f"📊 详细结果请查看: {args.work_dir}")
        print("\n💡 提示:")
        print("- 温度参数影响结果稳定性，建议多次测试")
        print("- IQuiz 包含 40 道 IQ 题和 80 道 EQ 题")
        print("- 可以调整 temperature 参数获得更稳定的结果")
    else:
        print("\n❌ 评测失败！")
        print("请检查:")
        print("1. vLLM 服务是否正常运行")
        print("2. 模型是否正确加载")
        print("3. 网络连接是否正常")


if __name__ == "__main__":
    main()
