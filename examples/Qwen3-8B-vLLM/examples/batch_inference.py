"""
批量推理示例

演示如何使用 vLLM 进行高效的批量推理
"""

import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def get_model_path() -> str:
    """获取模型路径"""
    # 优先检查 AutoDL 环境
    autodl_path = "/root/autodl-tmp/Qwen/Qwen3-8B"
    if os.path.exists(autodl_path):
        return autodl_path
    
    # 检查本地 models 目录
    local_path = "../models/Qwen/Qwen3-8B"
    if os.path.exists(local_path):
        return local_path
    
    # 检查上级目录的 models
    parent_path = "./models/Qwen/Qwen3-8B"
    if os.path.exists(parent_path):
        return parent_path
    
    print("⚠️  未找到模型文件，请先运行 model_download.py 下载模型")
    return local_path


class BatchInferenceEngine:
    """批量推理引擎"""
    
    def __init__(self, model_path: str = None, max_model_len: int = 8192):
        """
        初始化批量推理引擎
        
        Args:
            model_path: 模型路径
            max_model_len: 模型最大长度
        """
        self.model_path = model_path or get_model_path()
        self.max_model_len = max_model_len
        
        # 设置环境变量
        os.environ['VLLM_USE_MODELSCOPE'] = 'True'
        
        # 加载分词器
        print(f"📁 加载分词器: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            use_fast=False
        )
        
        # 初始化 vLLM 引擎
        print(f"🚀 初始化 vLLM 引擎...")
        self.llm = LLM(
            model=self.model_path,
            max_model_len=self.max_model_len,
            trust_remote_code=True
        )
        
        print(f"✅ 批量推理引擎初始化完成")
    
    def prepare_prompts(
        self, 
        inputs: List[Dict[str, Any]], 
        enable_thinking: bool = False
    ) -> List[str]:
        """
        准备批量推理的提示词
        
        Args:
            inputs: 输入数据列表，每个元素包含 messages 或 prompt
            enable_thinking: 是否启用思考模式
        
        Returns:
            List[str]: 格式化后的提示词列表
        """
        prompts = []
        
        for input_data in inputs:
            if "messages" in input_data:
                # 对话格式
                prompt = self.tokenizer.apply_chat_template(
                    input_data["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )
            elif "prompt" in input_data:
                # 直接提示词格式
                prompt = input_data["prompt"]
            else:
                raise ValueError("输入数据必须包含 'messages' 或 'prompt' 字段")
            
            prompts.append(prompt)
        
        return prompts
    
    def batch_generate(
        self,
        inputs: List[Dict[str, Any]],
        enable_thinking: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        max_tokens: int = 2048
    ) -> List[Dict[str, Any]]:
        """
        批量生成文本
        
        Args:
            inputs: 输入数据列表
            enable_thinking: 是否启用思考模式
            temperature: 温度参数
            top_p: 核心采样概率
            top_k: 候选词数量限制
            max_tokens: 最大生成长度
        
        Returns:
            List[Dict[str, Any]]: 生成结果列表
        """
        # 准备提示词
        prompts = self.prepare_prompts(inputs, enable_thinking)
        
        # 设置采样参数
        if enable_thinking:
            # 思考模式推荐参数
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0,
                max_tokens=max_tokens,
                stop_token_ids=[151645, 151643]
            )
        else:
            # 普通模式参数
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                stop_token_ids=[151645, 151643]
            )
        
        # 记录开始时间
        start_time = time.time()
        
        # 批量生成
        print(f"⏳ 开始批量推理 ({len(prompts)} 个请求)...")
        outputs = self.llm.generate(prompts, sampling_params)
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ 批量推理完成，耗时: {duration:.2f} 秒")
        print(f"📊 平均每个请求: {duration/len(prompts):.2f} 秒")
        
        # 处理结果
        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            
            # 解析思考模式响应
            thinking_content = ""
            final_answer = generated_text
            
            if enable_thinking and "<think>" in generated_text and "</think>" in generated_text:
                start_idx = generated_text.find("<think>") + 7
                end_idx = generated_text.find("</think>")
                thinking_content = generated_text[start_idx:end_idx].strip()
                final_answer = generated_text[end_idx + 8:].strip()
            elif "<think>" in generated_text and "</think>" in generated_text:
                # 清理空思考标签
                start_idx = generated_text.find("</think>") + 8
                final_answer = generated_text[start_idx:].strip()
            
            result = {
                "index": i,
                "input": inputs[i],
                "output": final_answer,
                "thinking": thinking_content,
                "raw_output": generated_text,
                "finish_reason": output.outputs[0].finish_reason
            }
            results.append(result)
        
        return results


def demo_qa_batch():
    """问答批量推理演示"""
    print("\n📚 问答批量推理演示")
    print("=" * 40)
    
    # 初始化推理引擎
    engine = BatchInferenceEngine()
    
    # 准备问答数据
    qa_inputs = [
        {
            "id": "qa_1",
            "messages": [
                {"role": "user", "content": "什么是机器学习？"}
            ]
        },
        {
            "id": "qa_2", 
            "messages": [
                {"role": "user", "content": "深度学习和机器学习有什么区别？"}
            ]
        },
        {
            "id": "qa_3",
            "messages": [
                {"role": "user", "content": "什么是神经网络？"}
            ]
        },
        {
            "id": "qa_4",
            "messages": [
                {"role": "user", "content": "解释一下什么是过拟合？"}
            ]
        },
        {
            "id": "qa_5",
            "messages": [
                {"role": "user", "content": "什么是梯度下降？"}
            ]
        }
    ]
    
    # 批量推理
    results = engine.batch_generate(qa_inputs, enable_thinking=False)
    
    # 显示结果
    for result in results:
        print(f"\n📝 问题 {result['index'] + 1}: {result['input']['messages'][0]['content']}")
        print(f"🤖 回答: {result['output'][:200]}...")
        print(f"🏁 结束原因: {result['finish_reason']}")


def demo_code_generation_batch():
    """代码生成批量推理演示"""
    print("\n💻 代码生成批量推理演示")
    print("=" * 40)
    
    # 初始化推理引擎
    engine = BatchInferenceEngine()
    
    # 准备代码生成任务
    code_inputs = [
        {
            "task": "排序算法",
            "messages": [
                {"role": "user", "content": "写一个 Python 函数实现快速排序算法"}
            ]
        },
        {
            "task": "数据结构",
            "messages": [
                {"role": "user", "content": "实现一个简单的链表类"}
            ]
        },
        {
            "task": "算法题",
            "messages": [
                {"role": "user", "content": "写一个函数判断一个字符串是否是回文"}
            ]
        },
        {
            "task": "工具函数",
            "messages": [
                {"role": "user", "content": "写一个函数计算两个数的最大公约数"}
            ]
        }
    ]
    
    # 批量推理
    results = engine.batch_generate(
        code_inputs, 
        enable_thinking=False,
        temperature=0.2,  # 代码生成使用较低温度
        max_tokens=1024
    )
    
    # 显示结果
    for result in results:
        print(f"\n💻 任务: {result['input']['task']}")
        print(f"📝 要求: {result['input']['messages'][0]['content']}")
        print(f"🔧 生成代码:")
        print(result['output'][:500] + "..." if len(result['output']) > 500 else result['output'])


def demo_thinking_batch():
    """思考模式批量推理演示"""
    print("\n🧠 思考模式批量推理演示")
    print("=" * 40)
    
    # 初始化推理引擎
    engine = BatchInferenceEngine()
    
    # 准备需要推理的问题
    thinking_inputs = [
        {
            "type": "数学",
            "messages": [
                {"role": "user", "content": "计算 7 的阶乘是多少？"}
            ]
        },
        {
            "type": "逻辑",
            "messages": [
                {"role": "user", "content": "如果所有的猫都是动物，而小花是一只猫，那么小花是动物吗？"}
            ]
        },
        {
            "type": "分析",
            "messages": [
                {"role": "user", "content": "为什么深度学习在图像识别方面如此成功？"}
            ]
        }
    ]
    
    # 批量推理（启用思考模式）
    results = engine.batch_generate(
        thinking_inputs, 
        enable_thinking=True,
        max_tokens=2048
    )
    
    # 显示结果
    for result in results:
        print(f"\n🔍 类型: {result['input']['type']}")
        print(f"📝 问题: {result['input']['messages'][0]['content']}")
        
        if result['thinking']:
            print(f"💭 思考过程: {result['thinking'][:300]}...")
            print("-" * 30)
        
        print(f"✅ 最终答案: {result['output']}")


def demo_performance_comparison():
    """性能对比演示"""
    print("\n⚡ 性能对比演示")
    print("=" * 40)
    
    # 初始化推理引擎
    engine = BatchInferenceEngine()
    
    # 准备测试数据
    test_inputs = [
        {
            "messages": [
                {"role": "user", "content": f"请简单介绍一下人工智能的第 {i+1} 个应用领域。"}
            ]
        }
        for i in range(10)
    ]
    
    print(f"📊 测试数据量: {len(test_inputs)} 个请求")
    
    # 批量推理
    print("\n🚀 批量推理:")
    start_time = time.time()
    batch_results = engine.batch_generate(
        test_inputs,
        enable_thinking=False,
        temperature=0.7,
        max_tokens=512
    )
    batch_time = time.time() - start_time
    
    print(f"⏱️  批量推理总时间: {batch_time:.2f} 秒")
    print(f"📈 平均每个请求: {batch_time/len(test_inputs):.2f} 秒")
    print(f"🔥 吞吐量: {len(test_inputs)/batch_time:.2f} 请求/秒")
    
    # 显示部分结果
    print(f"\n📋 部分结果预览:")
    for i, result in enumerate(batch_results[:3]):
        print(f"  {i+1}. {result['output'][:100]}...")


def save_results_to_file(results: List[Dict[str, Any]], filename: str):
    """保存结果到文件"""
    output_dir = Path("batch_results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / filename
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 结果已保存到: {output_file}")


def main():
    """主函数"""
    print("🚀 Qwen3-8B 批量推理示例")
    print("=" * 50)
    
    try:
        # 运行各种批量推理演示
        demo_qa_batch()
        demo_code_generation_batch()
        demo_thinking_batch()
        demo_performance_comparison()
        
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
    
    print("\n🎉 批量推理示例完成！")


if __name__ == "__main__":
    main()
