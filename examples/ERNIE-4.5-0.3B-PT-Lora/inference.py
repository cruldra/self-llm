"""
ERNIE-4.5-0.3B-PT LoRA 推理脚本

加载训练好的 LoRA 权重进行推理测试
"""

import os

import torch
from config import InferenceConfig, Paths
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_path, lora_path):
    """加载模型和 tokenizer"""
    print(f"加载模型: {model_path}")
    print(f"加载 LoRA 权重: {lora_path}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto",
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    
    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(model, model_id=lora_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, system_message=None):
    """生成回复"""
    if system_message is None:
        system_message = InferenceConfig.SYSTEM_MESSAGE

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    # 应用 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码输入
    model_inputs = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(model.device)

    # 生成回复
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=InferenceConfig.MAX_NEW_TOKENS,
            do_sample=InferenceConfig.DO_SAMPLE,
            temperature=InferenceConfig.TEMPERATURE,
            top_p=InferenceConfig.TOP_P,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 解码输出
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return response

def main():
    """主推理函数"""

    # 查找最新的 checkpoint
    checkpoints = []
    if os.path.exists(Paths.OUTPUT_DIR):
        for item in os.listdir(Paths.OUTPUT_DIR):
            if item.startswith("checkpoint-"):
                checkpoints.append(item)

    if not checkpoints:
        print(f"未找到训练好的模型，请先运行 train.py")
        return

    # 使用最新的 checkpoint
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
    lora_path = os.path.join(Paths.OUTPUT_DIR, latest_checkpoint)

    print(f"使用 checkpoint: {latest_checkpoint}")

    # 加载模型
    try:
        model, tokenizer = load_model_and_tokenizer(Paths.MODEL_PATH, lora_path)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 测试对话
    test_prompts = [
        "你是谁？",
        "你的家人都有谁？",
        "你最喜欢什么？",
        "你对皇上有什么看法？",
        "你在宫中的生活如何？"
    ]
    
    print("\n开始测试对话...")
    print("=" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n测试 {i}: {prompt}")
        print("-" * 30)
        
        try:
            response = generate_response(model, tokenizer, prompt)
            print(f"甄嬛: {response}")
        except Exception as e:
            print(f"生成回复失败: {e}")
        
        print("-" * 30)
    
    # 交互式对话
    print("\n进入交互式对话模式（输入 'quit' 退出）:")
    print("=" * 50)
    
    while True:
        user_input = input("\n你: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("再见！")
            break
        
        if not user_input:
            continue
        
        try:
            response = generate_response(model, tokenizer, user_input)
            print(f"甄嬛: {response}")
        except Exception as e:
            print(f"生成回复失败: {e}")

if __name__ == "__main__":
    main()
