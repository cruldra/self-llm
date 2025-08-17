"""
流式对话示例

演示如何使用 OpenAI API 进行流式对话
需要先启动 vLLM API 服务器
"""

import asyncio
import time
from openai import OpenAI, AsyncOpenAI
from typing import List, Dict, Any, AsyncGenerator


class StreamingChatClient:
    """流式对话客户端"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "sk-xxx",
        model: str = "Qwen3-8B"
    ):
        """
        初始化流式对话客户端
        
        Args:
            base_url: API 服务器地址
            api_key: API 密钥 (随便填写)
            model: 模型名称
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        
        # 同步客户端
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        # 异步客户端
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        print(f"🌐 连接到服务器: {base_url}")
        print(f"🏷️  使用模型: {model}")
    
    def check_server_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            models = self.client.models.list()
            print(f"✅ 服务器运行正常，可用模型: {[m.id for m.id in models.data]}")
            return True
        except Exception as e:
            print(f"❌ 服务器连接失败: {e}")
            print("💡 请先启动服务器:")
            print("   uv run python start_api_server.py")
            return False
    
    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        同步流式对话
        
        Args:
            messages: 消息历史
            temperature: 温度参数
            max_tokens: 最大生成长度
        
        Returns:
            str: 完整的回复内容
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            return full_response
            
        except Exception as e:
            print(f"❌ 流式对话失败: {e}")
            return ""
    
    async def async_stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        异步流式对话
        
        Args:
            messages: 消息历史
            temperature: 温度参数
            max_tokens: 最大生成长度
        
        Returns:
            str: 完整的回复内容
        """
        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            full_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            return full_response
            
        except Exception as e:
            print(f"❌ 异步流式对话失败: {e}")
            return ""


def demo_basic_streaming():
    """基础流式对话演示"""
    print("\n💬 基础流式对话演示")
    print("=" * 40)
    
    # 初始化客户端
    client = StreamingChatClient()
    
    # 检查服务器状态
    if not client.check_server_health():
        return
    
    # 测试问题
    questions = [
        "请详细介绍一下什么是深度学习？",
        "深度学习在计算机视觉领域有哪些应用？",
        "写一个 Python 函数来实现二分查找算法。"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 问题 {i}: {question}")
        print("🤖 助手: ", end="")
        
        messages = [{"role": "user", "content": question}]
        
        start_time = time.time()
        response = client.stream_chat(messages, temperature=0.7)
        end_time = time.time()
        
        print(f"\n⏱️  响应时间: {end_time - start_time:.2f} 秒")
        print(f"📏 回复长度: {len(response)} 字符")


def demo_multi_turn_streaming():
    """多轮流式对话演示"""
    print("\n🔄 多轮流式对话演示")
    print("=" * 40)
    
    # 初始化客户端
    client = StreamingChatClient()
    
    # 检查服务器状态
    if not client.check_server_health():
        return
    
    # 对话历史
    conversation = []
    
    # 模拟多轮对话
    turns = [
        "我想学习机器学习，应该从哪里开始？",
        "我已经有一些 Python 基础了，还需要学习什么？",
        "推荐一些实践项目吧。",
        "谢谢你的建议！有什么在线课程推荐吗？"
    ]
    
    for i, user_input in enumerate(turns, 1):
        print(f"\n💬 第 {i} 轮对话")
        print(f"👤 用户: {user_input}")
        print("🤖 助手: ", end="")
        
        # 添加用户消息
        conversation.append({"role": "user", "content": user_input})
        
        # 流式生成回复
        response = client.stream_chat(conversation.copy(), temperature=0.7)
        
        # 添加助手回复到历史
        conversation.append({"role": "assistant", "content": response})
        
        print()  # 换行


async def demo_async_streaming():
    """异步流式对话演示"""
    print("\n⚡ 异步流式对话演示")
    print("=" * 40)
    
    # 初始化客户端
    client = StreamingChatClient()
    
    # 检查服务器状态
    if not client.check_server_health():
        return
    
    # 并发处理多个问题
    questions = [
        "什么是神经网络？",
        "解释一下什么是过拟合？",
        "什么是梯度下降算法？"
    ]
    
    async def process_question(question: str, index: int):
        """处理单个问题"""
        print(f"\n📝 问题 {index}: {question}")
        print(f"🤖 助手 {index}: ", end="")
        
        messages = [{"role": "user", "content": question}]
        
        start_time = time.time()
        response = await client.async_stream_chat(messages, temperature=0.7)
        end_time = time.time()
        
        print(f"\n⏱️  问题 {index} 响应时间: {end_time - start_time:.2f} 秒")
        return response
    
    # 并发执行
    print("🚀 开始并发处理...")
    start_time = time.time()
    
    tasks = [
        process_question(question, i+1) 
        for i, question in enumerate(questions)
    ]
    
    responses = await asyncio.gather(*tasks)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n📊 总处理时间: {total_time:.2f} 秒")
    print(f"📈 平均每个问题: {total_time/len(questions):.2f} 秒")


def demo_interactive_streaming():
    """交互式流式对话"""
    print("\n💬 交互式流式对话")
    print("=" * 40)
    print("输入 'quit' 退出，输入 'clear' 清空历史")
    
    # 初始化客户端
    client = StreamingChatClient()
    
    # 检查服务器状态
    if not client.check_server_health():
        return
    
    conversation = []
    
    while True:
        try:
            user_input = input("\n👤 用户: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 再见！")
                break
            elif user_input.lower() in ['clear', '清空']:
                conversation = []
                print("🗑️  对话历史已清空")
                continue
            elif not user_input:
                continue
            
            # 添加用户消息
            conversation.append({"role": "user", "content": user_input})
            
            # 流式生成回复
            print("🤖 助手: ", end="")
            response = client.stream_chat(conversation.copy(), temperature=0.7)
            
            # 添加助手回复到历史
            conversation.append({"role": "assistant", "content": response})
            
            # 限制历史长度
            if len(conversation) > 20:
                conversation = conversation[-20:]
                
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 对话过程中出现错误: {e}")


def demo_streaming_with_thinking():
    """带思考过程的流式对话"""
    print("\n🧠 带思考过程的流式对话")
    print("=" * 40)
    
    # 初始化客户端
    client = StreamingChatClient()
    
    # 检查服务器状态
    if not client.check_server_health():
        return
    
    # 需要推理的问题
    questions = [
        "计算 8 的阶乘，并详细说明计算过程。<think>\n",
        "如果一个圆的半径是 5，那么它的面积和周长分别是多少？<think>\n"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 问题 {i}: {question.replace('<think>', '').strip()}")
        print("🤖 助手 (包含思考过程): ", end="")
        
        messages = [{"role": "user", "content": question}]
        
        response = client.stream_chat(messages, temperature=0.6, max_tokens=2048)
        print()


def main():
    """主函数"""
    print("🌊 Qwen3-8B 流式对话示例")
    print("=" * 50)
    
    try:
        # 运行各种流式对话演示
        demo_basic_streaming()
        demo_multi_turn_streaming()
        
        # 异步演示
        print("\n🔄 运行异步演示...")
        asyncio.run(demo_async_streaming())
        
        # 思考模式演示
        demo_streaming_with_thinking()
        
        # 询问是否进入交互模式
        choice = input("\n❓ 是否进入交互式流式对话模式？(y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            demo_interactive_streaming()
            
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
    
    print("\n🎉 流式对话示例完成！")


if __name__ == "__main__":
    main()
