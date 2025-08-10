"""
ERNIE-4.5-0.3B-PT LoRA 微调训练脚本

基于甄嬛数据集对 ERNIE-4.5-0.3B-PT 模型进行 LoRA 微调
使用 SwanLab 进行训练过程可视化
"""

import json
import os

import swanlab
import torch
from config import LoRAConfig, Paths, SwanLabConfig, TrainingConfig
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from swanlab.integration.transformers import SwanLabCallback
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def load_dataset(data_path):
    """加载甄嬛数据集"""
    print(f"加载数据集: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"数据集大小: {len(data)}")
    return Dataset.from_list(data)

def process_func(example, tokenizer):
    """数据预处理函数"""
    input_ids, attention_mask, labels = [], [], []

    # 适配 chat_template
    instruction = tokenizer(
        f"<|begin_of_sentence|>现在你要扮演皇帝身边的女人--甄嬛\n"
        f"User: {example['instruction']}\n"
        f"Assistant: ",
        add_special_tokens=False
    )
    response = tokenizer(f"{example['output']}<|end_of_sentence|>", add_special_tokens=False)

    # 拼接 input_ids
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    # 注意力掩码
    attention_mask = [1] * len(input_ids)
    # 标签，instruction 部分使用 -100 表示不计算 loss
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    # 截断处理
    if len(input_ids) > TrainingConfig.MAX_LENGTH:
        input_ids = input_ids[:TrainingConfig.MAX_LENGTH]
        attention_mask = attention_mask[:TrainingConfig.MAX_LENGTH]
        labels = labels[:TrainingConfig.MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main():
    """主训练函数"""

    # 检查模型路径
    if not os.path.exists(Paths.MODEL_PATH):
        print(f"模型路径不存在: {Paths.MODEL_PATH}")
        print("请先运行 model_download.py 下载模型")
        return

    # 检查数据集路径
    if not os.path.exists(Paths.DATASET_PATH):
        print(f"数据集路径不存在: {Paths.DATASET_PATH}")
        return
    
    print("开始加载模型和数据...")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Paths.MODEL_PATH,
        use_fast=False,
        trust_remote_code=True
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        Paths.MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 启用梯度检查点
    model.enable_input_require_grads()

    # 加载数据集
    dataset = load_dataset(Paths.DATASET_PATH)
    
    # 数据预处理
    print("开始数据预处理...")
    tokenized_dataset = dataset.map(
        lambda example: process_func(example, tokenizer),
        remove_columns=dataset.column_names
    )
    
    # 配置 LoRA
    print("配置 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=LoRAConfig.TARGET_MODULES,
        inference_mode=False,
        r=LoRAConfig.R,
        lora_alpha=LoRAConfig.LORA_ALPHA,
        lora_dropout=LoRAConfig.LORA_DROPOUT
    )

    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 训练参数
    training_args = TrainingArguments(
        output_dir=Paths.OUTPUT_DIR,
        per_device_train_batch_size=TrainingConfig.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=TrainingConfig.GRADIENT_ACCUMULATION_STEPS,
        logging_steps=TrainingConfig.LOGGING_STEPS,
        num_train_epochs=TrainingConfig.NUM_TRAIN_EPOCHS,
        save_steps=TrainingConfig.SAVE_STEPS,
        learning_rate=TrainingConfig.LEARNING_RATE,
        save_on_each_node=True,
        gradient_checkpointing=TrainingConfig.GRADIENT_CHECKPOINTING,
        report_to="none",
    )
    
    # SwanLab 回调
    swanlab_callback = SwanLabCallback(
        project=SwanLabConfig.PROJECT_NAME,
        experiment_name=SwanLabConfig.EXPERIMENT_NAME
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback]
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存模型
    print("保存模型...")
    trainer.save_model()
    
    print("训练完成！")

if __name__ == "__main__":
    main()
