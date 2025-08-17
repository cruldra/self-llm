"""
Qwen3-8B vLLM 基础推理示例

演示如何使用 vLLM 进行基础的文本生成推理
这是最简单的使用示例，适合初学者
"""

import os
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


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


def basic_text_generation():
    """基础文本生成示例"""
    print("🚀 基础文本生成示例")
    print("=" * 40)
    
    # 设置环境变量
    os.environ['VLLM_USE_MODELSCOPE'] = 'True'
    
    # 获取模型路径
    model_path = get_model_path()
    print(f"📁 模型路径: {model_path}")
    
    # 初始化 vLLM 引擎
    print("⏳ 初始化 vLLM 引擎...")
    llm = LLM(
        model=model_path,
        max_model_len=8192,
        trust_remote_code=True
    )
    print("✅ vLLM 引擎初始化完成")
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=512,
        stop_token_ids=[151645, 151643]
    )
    
    # 测试提示词
    prompts = [
        "人工智能是",
        "深度学习的主要优势包括",
        "Python 是一种",
        "机器学习在医疗领域的应用有"
    ]
    
    print("\n📝 开始文本生成...")
    
    # 生成文本
    outputs = llm.generate(prompts, sampling_params)
    
    # 显示结果
    for i, output in enumerate(outputs):
        prompt = prompts[i]
        generated_text = output.outputs[0].text
        
        print(f"\n🔍 提示词 {i+1}: {prompt}")
        print(f"🤖 生成文本: {generated_text}")
        print(f"🏁 结束原因: {output.outputs[0].finish_reason}")
        print("-" * 40)


def chat_format_generation():
    """对话格式生成示例"""
    print("\n💬 对话格式生成示例")
    print("=" * 40)
    
    # 设置环境变量
    os.environ['VLLM_USE_MODELSCOPE'] = 'True'
    
    # 获取模型路径
    model_path = get_model_path()
    
    # 加载分词器
    print("📁 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # 初始化 vLLM 引擎
    print("⏳ 初始化 vLLM 引擎...")
    llm = LLM(
        model=model_path,
        max_model_len=8192,
        trust_remote_code=True
    )
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=1024,
        stop_token_ids=[151645, 151643]
    )
    
    # 准备对话数据
    conversations = [
        [
            {"role": "user", "content": "你好，你是谁？"}
        ],
        [
            {"role": "user", "content": "什么是机器学习？"}
        ],
        [
            {"role": "user", "content": "写一个 Python 函数来计算斐波那契数列"}
        ]
    ]
    
    # 格式化对话为模型输入
    prompts = []
    for messages in conversations:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # 禁用思考模式
        )
        prompts.append(prompt)
    
    print("📝 开始对话生成...")
    
    # 生成回复
    outputs = llm.generate(prompts, sampling_params)
    
    # 显示结果
    for i, output in enumerate(outputs):
        user_message = conversations[i][0]["content"]
        generated_text = output.outputs[0].text
        
        print(f"\n👤 用户: {user_message}")
        print(f"🤖 助手: {generated_text}")
        print(f"🏁 结束原因: {output.outputs[0].finish_reason}")
        print("-" * 40)


def parameter_comparison():
    """参数对比示例"""
    print("\n⚙️  参数对比示例")
    print("=" * 40)
    
    # 设置环境变量
    os.environ['VLLM_USE_MODELSCOPE'] = 'True'
    
    # 获取模型路径
    model_path = get_model_path()
    
    # 初始化 vLLM 引擎
    print("⏳ 初始化 vLLM 引擎...")
    llm = LLM(
        model=model_path,
        max_model_len=8192,
        trust_remote_code=True
    )
    
    # 测试提示词
    prompt = "写一个关于春天的短诗："
    
    # 不同的采样参数配置
    param_configs = [
        {
            "name": "保守模式",
            "params": SamplingParams(
                temperature=0.3,
                top_p=0.8,
                max_tokens=200,
                stop_token_ids=[151645, 151643]
            )
        },
        {
            "name": "平衡模式", 
            "params": SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=200,
                stop_token_ids=[151645, 151643]
            )
        },
        {
            "name": "创意模式",
            "params": SamplingParams(
                temperature=1.0,
                top_p=0.95,
                max_tokens=200,
                stop_token_ids=[151645, 151643]
            )
        }
    ]
    
    print(f"📝 测试提示词: {prompt}")
    print("\n🔄 使用不同参数生成...")
    
    for config in param_configs:
        print(f"\n🎛️  {config['name']}:")
        print(f"   Temperature: {config['params'].temperature}")
        print(f"   Top-p: {config['params'].top_p}")
        
        # 生成文本
        outputs = llm.generate([prompt], config['params'])
        generated_text = outputs[0].outputs[0].text
        
        print(f"🤖 生成结果: {generated_text}")
        print("-" * 30)


def simple_qa_demo():
    """简单问答演示"""
    print("\n❓ 简单问答演示")
    print("=" * 40)
    
    # 设置环境变量
    os.environ['VLLM_USE_MODELSCOPE'] = 'True'
    
    # 获取模型路径
    model_path = get_model_path()
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # 初始化 vLLM 引擎
    llm = LLM(
        model=model_path,
        max_model_len=8192,
        trust_remote_code=True
    )
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
        stop_token_ids=[151645, 151643]
    )
    
    # 问答对
    qa_pairs = [
        "什么是深度学习？",
        "Python 有什么优势？",
        "如何学习机器学习？",
        "什么是神经网络？"
    ]
    
    for i, question in enumerate(qa_pairs, 1):
        print(f"\n📝 问题 {i}: {question}")
        
        # 格式化为对话格式
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # 生成回答
        outputs = llm.generate([prompt], sampling_params)
        answer = outputs[0].outputs[0].text
        
        print(f"🤖 回答: {answer}")
        print("-" * 30)


def main():
    """主函数"""
    print("🤖 Qwen3-8B vLLM 基础推理示例")
    print("=" * 50)
    
    try:
        # 运行各种基础示例
        basic_text_generation()
        chat_format_generation()
        parameter_comparison()
        simple_qa_demo()
        
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        print("💡 请确保:")
        print("   1. 模型已正确下载")
        print("   2. 有足够的 GPU 显存")
        print("   3. 依赖包已正确安装")
    
    print("\n🎉 基础推理示例完成！")
    print("💡 接下来可以尝试:")
    print("   - uv run python vllm_thinking_mode.py")
    print("   - uv run python vllm_non_thinking_mode.py")
    print("   - uv run python examples/basic_chat.py")


if __name__ == "__main__":
    main()
