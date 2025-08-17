"""
OpenAI Completions API 测试脚本

测试 vLLM 启动的 OpenAI 兼容 API 服务器的 Completions 接口
"""

import requests
import json
from openai import OpenAI
from typing import Dict, Any


def test_with_curl(
    base_url: str = "http://192.168.1.2:8000",
    model: str = "Qwen3-8B"
) -> None:
    """
    使用 requests 库测试 Completions API (模拟 curl)
    
    Args:
        base_url: API 服务器地址
        model: 模型名称
    """
    print("🔧 使用 requests 测试 Completions API")
    print("-" * 40)
    
    url = f"{base_url}/v1/completions"
    headers = {
        "Content-Type": "application/json"
    }
    
    # 测试数据
    test_cases = [
        {
            "name": "数学计算",
            "prompt": "我想问你，5的阶乘是多少？<think>\n",
            "max_tokens": 1024,
            "temperature": 0
        },
        {
            "name": "简单问答",
            "prompt": "什么是人工智能？",
            "max_tokens": 512,
            "temperature": 0.7
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 测试用例 {i}: {test_case['name']}")
        
        data = {
            "model": model,
            "prompt": test_case["prompt"],
            "max_tokens": test_case["max_tokens"],
            "temperature": test_case["temperature"]
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            print(f"✅ 请求成功")
            print(f"📊 状态码: {response.status_code}")
            print(f"🆔 请求ID: {result.get('id', 'N/A')}")
            print(f"📝 生成文本: {result['choices'][0]['text'][:200]}...")
            print(f"📈 Token 使用: {result.get('usage', {})}")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败: {e}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失败: {e}")
        except Exception as e:
            print(f"❌ 其他错误: {e}")


def test_with_openai_client(
    base_url: str = "http://192.168.1.2:8000/v1",
    api_key: str = "sk-xxx",
    model: str = "Qwen3-8B"
) -> None:
    """
    使用 OpenAI 客户端测试 Completions API
    
    Args:
        base_url: API 服务器地址
        api_key: API 密钥 (随便填写)
        model: 模型名称
    """
    print("\n🔧 使用 OpenAI 客户端测试 Completions API")
    print("-" * 40)
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    # 测试数据
    test_cases = [
        {
            "name": "代码生成",
            "prompt": "写一个 Python 函数来计算斐波那契数列：",
            "max_tokens": 512,
            "temperature": 0.3
        },
        {
            "name": "创意写作",
            "prompt": "写一个关于机器人的短故事：",
            "max_tokens": 800,
            "temperature": 0.8
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 测试用例 {i}: {test_case['name']}")
        
        try:
            completion = client.completions.create(
                model=model,
                prompt=test_case["prompt"],
                max_tokens=test_case["max_tokens"],
                temperature=test_case["temperature"]
            )
            
            print(f"✅ 请求成功")
            print(f"🆔 请求ID: {completion.id}")
            print(f"📝 生成文本: {completion.choices[0].text[:200]}...")
            print(f"📈 Token 使用: {completion.usage}")
            print(f"🏁 结束原因: {completion.choices[0].finish_reason}")
            
        except Exception as e:
            print(f"❌ 请求失败: {e}")


def test_models_endpoint(base_url: str = "http://192.168.1.2:8000") -> None:
    """
    测试模型列表接口
    
    Args:
        base_url: API 服务器地址
    """
    print("\n🔧 测试模型列表接口")
    print("-" * 40)
    
    url = f"{base_url}/v1/models"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        
        print(f"✅ 请求成功")
        print(f"📊 状态码: {response.status_code}")
        print(f"📋 可用模型:")
        
        for model in result.get("data", []):
            print(f"  - {model.get('id', 'Unknown')}")
            print(f"    创建时间: {model.get('created', 'Unknown')}")
            print(f"    最大长度: {model.get('max_model_len', 'Unknown')}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")


def check_server_health(base_url: str = "http://192.168.1.2:8000") -> bool:
    """
    检查服务器健康状态
    
    Args:
        base_url: API 服务器地址
    
    Returns:
        bool: 服务器是否健康
    """
    print("🏥 检查服务器健康状态")
    print("-" * 40)
    
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ 服务器运行正常")
            return True
        else:
            print(f"⚠️  服务器响应异常: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器")
        print("💡 请确保服务器已启动:")
        print("   uv run python start_api_server.py")
        return False
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False


def main():
    """主函数"""
    print("🧪 OpenAI Completions API 测试")
    print("=" * 50)
    
    base_url = "http://192.168.1.2:8000"
    model = "Qwen3-8B"
    
    # 检查服务器健康状态
    if not check_server_health(base_url):
        return
    
    # 测试模型列表接口
    test_models_endpoint(base_url)
    
    # 使用 requests 测试
    test_with_curl(base_url, model)
    
    # 使用 OpenAI 客户端测试
    test_with_openai_client(f"{base_url}/v1", "sk-xxx", model)
    
    print("\n" + "=" * 50)
    print("🎉 Completions API 测试完成！")


if __name__ == "__main__":
    main()
