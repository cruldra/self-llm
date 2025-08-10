"""
ERNIE-4.5-0.3B-PT LoRA 微调配置文件

包含训练和推理的各种配置参数
"""

import os

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 路径配置
class Paths:
    # 模型路径
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "PaddlePaddle", "ERNIE-4___5-0___3B-PT")
    
    # 数据集路径
    DATASET_PATH = os.path.join(PROJECT_ROOT, "..", "..", "dataset", "huanhuan.json")
    
    # 输出路径
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "ERNIE-4.5-lora")
    
    # 模型下载缓存路径
    CACHE_DIR = os.path.join(PROJECT_ROOT, "models")

# LoRA 配置
class LoRAConfig:
    # LoRA 秩
    R = 8
    
    # LoRA Alpha
    LORA_ALPHA = 32
    
    # Dropout 比例
    LORA_DROPOUT = 0.1
    
    # 目标模块
    TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]

# 训练配置
class TrainingConfig:
    # 最大序列长度
    MAX_LENGTH = 1024
    
    # 批量大小
    PER_DEVICE_TRAIN_BATCH_SIZE = 4
    
    # 梯度累积步数
    GRADIENT_ACCUMULATION_STEPS = 4
    
    # 学习率
    LEARNING_RATE = 1e-4
    
    # 训练轮次
    NUM_TRAIN_EPOCHS = 3
    
    # 日志步数
    LOGGING_STEPS = 10
    
    # 保存步数
    SAVE_STEPS = 100
    
    # 是否启用梯度检查点
    GRADIENT_CHECKPOINTING = True

# 推理配置
class InferenceConfig:
    # 最大新生成 token 数
    MAX_NEW_TOKENS = 1024
    
    # 是否采样
    DO_SAMPLE = True
    
    # 温度参数
    TEMPERATURE = 0.7
    
    # Top-p 参数
    TOP_P = 0.9
    
    # 系统消息
    SYSTEM_MESSAGE = "假设你是皇帝身边的女人--甄嬛。"

# SwanLab 配置
class SwanLabConfig:
    # 项目名称
    PROJECT_NAME = "self-llm"
    
    # 实验名称
    EXPERIMENT_NAME = "ERNIE-4.5-0.3B-lora"

# 模型下载配置
class ModelDownloadConfig:
    # 模型名称
    MODEL_NAME = "PaddlePaddle/ERNIE-4.5-0.3B-PT"
    
    # 版本
    REVISION = "master"
