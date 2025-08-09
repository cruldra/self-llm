# api.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, validator
from typing import List, Literal, Union
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import time
import logging
import uvicorn

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局模型变量

# 设置设备参数
DEVICE = "cuda"  # 使用CUDA
DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息

model = None
processor = None
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
print(torch.cuda.is_available())
# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片
            
# --- Pydantic模型定义 ---
class ContentItem(BaseModel):
    type: Literal["text", "image"]
    text: str = Field(None, description="文本内容（当type为text时必填）")
    image: str = Field(None, description="图片URL或base64字符串（当type为image时必填）")

    @validator('*', pre=True)
    def validate_content(cls, v, values):
        if values.get('type') == 'text' and not v:
            raise ValueError("文本类型必须提供text字段")
        if values.get('type') == 'image':
            if not (v.startswith(('http://', 'https://', 'data:image'))):
                raise ValueError("图片必须是有效的URL或base64编码字符串")
        return v

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: List[ContentItem]

class ProcessRequest(BaseModel):
    messages: List[Message] = Field(..., min_items=1, description="对话历史记录")
    max_new_tokens: int = Field(1000, ge=10, le=4096, description="生成的最大token数")

class ProcessResponse(BaseModel):
    response: str
    status: int
    time: int
    processing_time: float
    tokens_generated: int

# --- 模型加载逻辑 ---
def load_models():
    global model, processor
    try:
        logger.info("正在加载模型...")
        model_id = "./models/LLM-Research/gemma-3-4b-it"
        
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        ).eval()
        
        logger.info("正在加载处理器...")
        processor = AutoProcessor.from_pretrained(model_id)
        logger.info("模型加载完成")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise

# --- FastAPI应用 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_models()
        yield
    except Exception as e:
        logger.error(f"服务初始化失败: {str(e)}")
        raise

app = FastAPI(lifespan=lifespan)

@app.post("/chat/completions", response_model=ProcessResponse)
async def generate_response(request: ProcessRequest):
    start_time = time.time()
    
    try:
        # 构建消息格式
        processed_messages = []
        system_prompt = DEFAULT_SYSTEM_PROMPT
        
        # 提取或替换系统提示
        for msg in request.messages:
            if msg.role == "system":
                system_prompt = " ".join([item.text for item in msg.content if item.type == "text"])
            else:
                processed_messages.append(msg.dict())

        # 构建最终消息结构并处理图像
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            *processed_messages
        ]
 
        # 准备模型输入
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(model.device, dtype=torch.bfloat16)

        # 生成响应
        input_len = inputs["input_ids"].shape[-1]

        max_token_num = 4096
        if max_token_num >= int(request.max_new_tokens):
            max_token_num = int(request.max_new_tokens)
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=max_token_num,
                do_sample=False
            )
        generated = generation[0][input_len:]

        # 解码结果
        decoded = processor.decode(generated, skip_special_tokens=True)

        return ProcessResponse(
            response=decoded,
            status=200,
            time=int(time.time()),
            processing_time=time.time() - start_time,
            tokens_generated=len(generated)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=6006)
