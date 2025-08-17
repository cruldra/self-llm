"""
基础对话示例

演示如何使用 vLLM 进行基础的对话交互
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
    local_path = "../models/Qwen/Qwen3-8B"
    if os.path.exists(local_path):
        return local_path
    
    # 检查上级目录的 models
    parent_path = "./models/Qwen/Qwen3-8B"
    if os.path.exists(parent_path):
        return parent_path
    
    print("⚠️  未找到模型文件，请先运行 model_download.py 下载模型")
    return local_path


class QwenChatBot:
    """Qwen 聊天机器人类"""
    
    def __init__(self, model_path: str = None, enable_thinking: bool = False):
        """
        初始化聊天机器人
        
        Args:
            model_path: 模型路径
            enable_thinking: 是否启用思考模式
        """
        self.model_path = model_path or get_model_path()
        self.enable_thinking = enable_thinking
        
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
            max_model_len=8192,
            trust_remote_code=True
        )
        
        # 设置采样参数
        if enable_thinking:
            # 思考模式推荐参数
            self.sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0,
                max_tokens=4096,
                stop_token_ids=[151645, 151643]
            )
        else:
            # 非思考模式推荐参数
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0,
                max_tokens=4096,
                stop_token_ids=[151645, 151643]
            )
        
        print(f"✅ 聊天机器人初始化完成 (思考模式: {'启用' if enable_thinking else '禁用'})")
    
    def format_messages(self, messages: list) -> str:
        """
        格式化消息为模型输入
        
        Args:
            messages: 消息列表
        
        Returns:
            str: 格式化后的输入文本
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
    
    def chat(self, user_input: str, conversation_history: list = None) -> tuple:
        """
        进行对话
        
        Args:
            user_input: 用户输入
            conversation_history: 对话历史
        
        Returns:
            tuple: (回复内容, 思考过程)
        """
        # 构建消息历史
        messages = conversation_history or []
        messages.append({"role": "user", "content": user_input})
        
        # 格式化输入
        prompt = self.format_messages(messages)
        
        # 生成回复
        outputs = self.llm.generate([prompt], self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # 解析响应
        if self.enable_thinking and "<think>" in generated_text and "</think>" in generated_text:
            # 分离思考过程和最终答案
            start_idx = generated_text.find("<think>") + 7
            end_idx = generated_text.find("</think>")
            thinking_content = generated_text[start_idx:end_idx].strip()
            final_answer = generated_text[end_idx + 8:].strip()
            return final_answer, thinking_content
        else:
            # 清理可能的空思考标签
            if "<think>" in generated_text and "</think>" in generated_text:
                start_idx = generated_text.find("</think>") + 8
                clean_text = generated_text[start_idx:].strip()
            else:
                clean_text = generated_text.strip()
            return clean_text, ""


def demo_single_turn():
    """单轮对话演示"""
    print("\n🔄 单轮对话演示")
    print("=" * 40)
    
    # 初始化聊天机器人
    chatbot = QwenChatBot(enable_thinking=False)
    
    # 测试问题
    questions = [
        "你好，你是谁？",
        "什么是人工智能？",
        "请推荐几本关于机器学习的书籍。",
        "写一个简单的 Python 函数来计算两个数的最大公约数。"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 问题 {i}: {question}")
        print("-" * 30)
        
        try:
            answer, _ = chatbot.chat(question)
            print(f"🤖 回答: {answer}")
        except Exception as e:
            print(f"❌ 错误: {e}")


def demo_multi_turn():
    """多轮对话演示"""
    print("\n🔄 多轮对话演示")
    print("=" * 40)
    
    # 初始化聊天机器人
    chatbot = QwenChatBot(enable_thinking=False)
    
    # 对话历史
    conversation = []
    
    # 模拟多轮对话
    turns = [
        "我想学习深度学习，应该从哪里开始？",
        "我已经有一些 Python 基础了，还需要学习什么数学知识？",
        "推荐一些实践项目吧。",
        "谢谢你的建议！"
    ]
    
    for i, user_input in enumerate(turns, 1):
        print(f"\n💬 第 {i} 轮对话")
        print(f"👤 用户: {user_input}")
        
        try:
            # 进行对话
            answer, _ = chatbot.chat(user_input, conversation.copy())
            print(f"🤖 助手: {answer}")
            
            # 更新对话历史
            conversation.append({"role": "user", "content": user_input})
            conversation.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            print(f"❌ 错误: {e}")


def demo_thinking_mode():
    """思考模式演示"""
    print("\n🧠 思考模式演示")
    print("=" * 40)
    
    # 初始化思考模式聊天机器人
    chatbot = QwenChatBot(enable_thinking=True)
    
    # 需要推理的问题
    questions = [
        "计算 6 的阶乘是多少？",
        "如果一个正方形的面积是 25，那么它的周长是多少？",
        "解释一下为什么深度学习在图像识别方面如此有效？"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 问题 {i}: {question}")
        print("-" * 30)
        
        try:
            answer, thinking = chatbot.chat(question)
            
            if thinking:
                print(f"💭 思考过程: {thinking[:200]}...")
                print("-" * 20)
            
            print(f"✅ 最终答案: {answer}")
            
        except Exception as e:
            print(f"❌ 错误: {e}")


def interactive_chat():
    """交互式对话"""
    print("\n💬 交互式对话模式")
    print("=" * 40)
    print("输入 'quit' 退出，输入 'clear' 清空历史，输入 'thinking' 切换思考模式")
    
    # 初始化聊天机器人
    enable_thinking = False
    chatbot = QwenChatBot(enable_thinking=enable_thinking)
    conversation = []
    
    while True:
        try:
            user_input = input(f"\n👤 用户 ({'思考模式' if enable_thinking else '普通模式'}): ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 再见！")
                break
            elif user_input.lower() in ['clear', '清空']:
                conversation = []
                print("🗑️  对话历史已清空")
                continue
            elif user_input.lower() in ['thinking', '思考']:
                enable_thinking = not enable_thinking
                chatbot = QwenChatBot(enable_thinking=enable_thinking)
                print(f"🔄 已切换到{'思考模式' if enable_thinking else '普通模式'}")
                continue
            elif not user_input:
                continue
            
            # 进行对话
            answer, thinking = chatbot.chat(user_input, conversation.copy())
            
            if thinking and enable_thinking:
                print(f"\n💭 思考过程: {thinking[:200]}...")
                print("-" * 20)
            
            print(f"\n🤖 助手: {answer}")
            
            # 更新对话历史
            conversation.append({"role": "user", "content": user_input})
            conversation.append({"role": "assistant", "content": answer})
            
            # 限制历史长度
            if len(conversation) > 20:
                conversation = conversation[-20:]
                
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 对话过程中出现错误: {e}")


def main():
    """主函数"""
    print("🤖 Qwen3-8B 基础对话示例")
    print("=" * 50)
    
    try:
        # 运行演示
        demo_single_turn()
        demo_multi_turn()
        demo_thinking_mode()
        
        # 询问是否进入交互模式
        choice = input("\n❓ 是否进入交互式对话模式？(y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            interactive_chat()
            
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
    
    print("\n🎉 基础对话示例完成！")


if __name__ == "__main__":
    main()
