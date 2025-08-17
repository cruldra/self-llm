"""
Qwen3-8B vLLM 非思考模式推理示例

演示如何使用 vLLM 进行 Qwen3-8B 模型的非思考模式推理
非思考模式直接给出答案，不显示推理过程
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


def create_non_thinking_prompt(user_input: str) -> str:
    """
    创建非思考模式的提示词
    
    Args:
        user_input: 用户输入
    
    Returns:
        str: 格式化后的提示词
    """
    messages = [
        {"role": "user", "content": user_input}
    ]
    
    # 获取模型路径并加载分词器
    model_path = get_model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # 应用聊天模板，禁用思考模式
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # 禁用思考模式
    )
    
    return text


def get_completion_non_thinking(
    prompts: list,
    model_path: str,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0,
    max_tokens: int = 4096,
    max_model_len: int = 8192
):
    """
    使用非思考模式进行文本生成
    
    Args:
        prompts: 提示词列表
        model_path: 模型路径
        temperature: 温度参数，控制生成文本的随机性
        top_p: 核心采样概率
        top_k: 候选词数量限制
        min_p: 最小概率阈值
        max_tokens: 最大生成长度
        max_model_len: 模型最大长度
    
    Returns:
        生成结果列表
    """
    # 设置停止词 ID
    stop_token_ids = [151645, 151643]
    
    # 创建采样参数 - 非思考模式推荐参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids
    )
    
    # 初始化 vLLM 推理引擎
    llm = LLM(
        model=model_path,
        max_model_len=max_model_len,
        trust_remote_code=True
    )
    
    # 生成文本
    outputs = llm.generate(prompts, sampling_params)
    return outputs


def clean_response(response_text: str) -> str:
    """
    清理响应文本，移除可能的思考标签
    
    Args:
        response_text: 原始响应文本
    
    Returns:
        str: 清理后的响应文本
    """
    # 移除可能存在的空思考标签
    if "<think>" in response_text and "</think>" in response_text:
        start_idx = response_text.find("</think>") + 8
        return response_text[start_idx:].strip()
    
    return response_text.strip()


def main():
    """主函数"""
    print("🤖 Qwen3-8B vLLM 非思考模式推理示例")
    print("=" * 50)
    
    # 设置环境变量
    os.environ['VLLM_USE_MODELSCOPE'] = 'True'
    
    # 获取模型路径
    model_path = get_model_path()
    print(f"📁 模型路径: {model_path}")
    
    # 测试问题列表
    test_questions = [
        "你是谁？",
        "今天天气怎么样？",
        "请用一句话介绍人工智能。",
        "写一个简单的 Python 函数来计算斐波那契数列。",
        "推荐几本关于机器学习的书籍。"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 测试问题 {i}: {question}")
        print("-" * 40)
        
        try:
            # 创建非思考模式提示词
            prompt = create_non_thinking_prompt(question)
            
            # 进行推理
            print("⏳ 正在生成回答...")
            outputs = get_completion_non_thinking([prompt], model_path)
            
            # 输出结果
            for output in outputs:
                generated_text = output.outputs[0].text
                clean_text = clean_response(generated_text)
                
                print(f"\n✅ 回答:")
                print(clean_text)
                
        except Exception as e:
            print(f"❌ 推理过程中出现错误: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 非思考模式推理示例完成！")
    
    # 交互式对话
    print("\n💬 进入交互式对话模式 (输入 'quit' 退出):")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\n👤 用户: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 再见！")
                break
            
            if not user_input:
                continue
            
            # 创建提示词并推理
            prompt = create_non_thinking_prompt(user_input)
            outputs = get_completion_non_thinking([prompt], model_path)
            
            # 输出回答
            for output in outputs:
                generated_text = output.outputs[0].text
                clean_text = clean_response(generated_text)
                print(f"\n🤖 助手: {clean_text}")
                
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 对话过程中出现错误: {e}")


if __name__ == "__main__":
    main()
