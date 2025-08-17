"""
OpenAI Chat Completions API 测试脚本

测试 vLLM 启动的 OpenAI 兼容 API 服务器的 Chat Completions 接口
"""

import requests
import json
from openai import OpenAI
from typing import List, Dict, Any


def test_with_curl(
    base_url: str = "http://localhost:8000",
    model: str = "Qwen3-8B"
) -> None:
    """
    使用 requests 库测试 Chat Completions API (模拟 curl)
    
    Args:
        base_url: API 服务器地址
        model: 模型名称
    """
    print("🔧 使用 requests 测试 Chat Completions API")
    print("-" * 40)
    
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    
    # 测试数据
    test_cases = [
        {
            "name": "数学推理",
            "messages": [
                {"role": "user", "content": "我想问你，5的阶乘是多少？<think>\n"}
            ],
            "temperature": 0.6
        },
        {
            "name": "简单对话",
            "messages": [
                {"role": "user", "content": "你好，你是谁？"}
            ],
            "temperature": 0.7
        },
        {
            "name": "多轮对话",
            "messages": [
                {"role": "user", "content": "什么是深度学习？"},
                {"role": "assistant", "content": "深度学习是机器学习的一个子领域..."},
                {"role": "user", "content": "它有什么应用？"}
            ],
            "temperature": 0.8
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 测试用例 {i}: {test_case['name']}")
        
        data = {
            "model": model,
            "messages": test_case["messages"],
            "temperature": test_case["temperature"],
            "max_tokens": 1024
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            print(f"✅ 请求成功")
            print(f"📊 状态码: {response.status_code}")
            print(f"🆔 请求ID: {result.get('id', 'N/A')}")
            
            # 显示回复内容
            choice = result['choices'][0]
            message = choice['message']
            
            print(f"📝 回复内容: {message['content'][:200]}...")
            
            # 如果有推理内容，也显示
            if 'reasoning_content' in message and message['reasoning_content']:
                print(f"💭 推理过程: {message['reasoning_content'][:100]}...")
            
            print(f"📈 Token 使用: {result.get('usage', {})}")
            print(f"🏁 结束原因: {choice.get('finish_reason', 'N/A')}")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败: {e}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失败: {e}")
        except Exception as e:
            print(f"❌ 其他错误: {e}")


def test_with_openai_client(
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "sk-xxx",
    model: str = "Qwen3-8B"
) -> None:
    """
    使用 OpenAI 客户端测试 Chat Completions API
    
    Args:
        base_url: API 服务器地址
        api_key: API 密钥 (随便填写)
        model: 模型名称
    """
    print("\n🔧 使用 OpenAI 客户端测试 Chat Completions API")
    print("-" * 40)
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    # 测试数据
    test_cases = [
        {
            "name": "技术问答",
            "messages": [
                {"role": "user", "content": "什么是 Transformer 架构？"}
            ],
            "temperature": 0.3
        },
        {
            "name": "代码生成",
            "messages": [
                {"role": "user", "content": "写一个 Python 函数来实现快速排序算法"}
            ],
            "temperature": 0.2
        },
        {
            "name": "创意写作",
            "messages": [
                {"role": "user", "content": "写一首关于春天的诗"}
            ],
            "temperature": 0.9
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 测试用例 {i}: {test_case['name']}")
        
        try:
            chat_completion = client.chat.completions.create(
                model=model,
                messages=test_case["messages"],
                temperature=test_case["temperature"],
                max_tokens=1024
            )
            
            print(f"✅ 请求成功")
            print(f"🆔 请求ID: {chat_completion.id}")
            
            # 显示回复内容
            message = chat_completion.choices[0].message
            print(f"📝 回复内容: {message.content[:200]}...")
            
            # 如果有推理内容，也显示
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                print(f"💭 推理过程: {message.reasoning_content[:100]}...")
            
            print(f"📈 Token 使用: {chat_completion.usage}")
            print(f"🏁 结束原因: {chat_completion.choices[0].finish_reason}")
            
        except Exception as e:
            print(f"❌ 请求失败: {e}")


def test_streaming_chat(
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "sk-xxx",
    model: str = "Qwen3-8B"
) -> None:
    """
    测试流式 Chat Completions API
    
    Args:
        base_url: API 服务器地址
        api_key: API 密钥
        model: 模型名称
    """
    print("\n🔧 测试流式 Chat Completions API")
    print("-" * 40)
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    messages = [
        {"role": "user", "content": "请详细解释什么是机器学习，包括其主要类型和应用。"}
    ]
    
    print("📝 问题: 请详细解释什么是机器学习，包括其主要类型和应用。")
    print("💬 流式回复:")
    
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print(f"\n\n✅ 流式响应完成")
        print(f"📏 总长度: {len(full_response)} 字符")
        
    except Exception as e:
        print(f"❌ 流式请求失败: {e}")


def interactive_chat(
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "sk-xxx",
    model: str = "Qwen3-8B"
) -> None:
    """
    交互式聊天测试
    
    Args:
        base_url: API 服务器地址
        api_key: API 密钥
        model: 模型名称
    """
    print("\n💬 交互式聊天测试 (输入 'quit' 退出)")
    print("-" * 40)
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    messages = []
    
    while True:
        try:
            user_input = input("\n👤 用户: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 再见！")
                break
            
            if not user_input:
                continue
            
            # 添加用户消息
            messages.append({"role": "user", "content": user_input})
            
            # 获取助手回复
            chat_completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
            
            assistant_message = chat_completion.choices[0].message.content
            print(f"\n🤖 助手: {assistant_message}")
            
            # 添加助手消息到历史
            messages.append({"role": "assistant", "content": assistant_message})
            
            # 限制历史长度
            if len(messages) > 10:
                messages = messages[-10:]
                
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 对话过程中出现错误: {e}")


def main():
    """主函数"""
    print("🧪 OpenAI Chat Completions API 测试")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    model = "Qwen3-8B"
    
    # 检查服务器健康状态
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code != 200:
            print("❌ 服务器未正常运行")
            print("💡 请先启动服务器:")
            print("   uv run python start_api_server.py")
            return
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器")
        print("💡 请先启动服务器:")
        print("   uv run python start_api_server.py")
        return
    
    print("✅ 服务器运行正常")
    
    # 使用 requests 测试
    test_with_curl(base_url, model)
    
    # 使用 OpenAI 客户端测试
    test_with_openai_client(f"{base_url}/v1", "sk-xxx", model)
    
    # 测试流式响应
    test_streaming_chat(f"{base_url}/v1", "sk-xxx", model)
    
    print("\n" + "=" * 50)
    print("🎉 Chat Completions API 测试完成！")
    
    # 询问是否进入交互模式
    try:
        choice = input("\n❓ 是否进入交互式聊天模式？(y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            interactive_chat(f"{base_url}/v1", "sk-xxx", model)
    except KeyboardInterrupt:
        print("\n👋 再见！")


if __name__ == "__main__":
    main()
