import requests
import json

def get_completion():
    headers = {'Content-Type': 'application/json'}
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "http://gips3.baidu.com/it/u=1821127123,1149655687&fm=3028&app=3028&f=JPEG&fmt=auto?w=720&h=1280"},
                    {"type": "text", "text": "请描述图片中的详情信息"}
                ]
            }
        ],
        "max_new_tokens": 4096
    }
    response = requests.post(url='http://127.0.0.1:6006/chat/completions', headers=headers, data=json.dumps(data))
    return response.json()['response']

def get_text_completion():
    """纯文本对话测试"""
    headers = {'Content-Type': 'application/json'}
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "帮我写一份科幻小说的大纲！"}
                ]
            }
        ],
        "max_new_tokens": 4096
    }
    response = requests.post(url='http://127.0.0.1:6006/chat/completions', headers=headers, data=json.dumps(data))
    return response.json()

if __name__ == '__main__':
    print("测试纯文本对话:")
    result = get_text_completion()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n" + "="*50 + "\n")
    
    print("测试图片+文本对话:")
    try:
        result = get_completion()
        print(result)
    except Exception as e:
        print(f"图片测试失败: {e}")
