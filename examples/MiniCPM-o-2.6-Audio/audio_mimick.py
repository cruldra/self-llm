"""
MiniCPM-o-2.6 语音模仿功能演示
"""
import torch
from modelscope import AutoModel, AutoTokenizer
import librosa
import os

class AudioMimickDemo:
    def __init__(self, model_path='./models/MiniCPM-o-2_6'):
        """
        初始化语音模仿演示
        
        Args:
            model_path (str): 模型路径
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """加载模型和分词器"""
        print("正在加载 MiniCPM-o-2.6 模型...")
        
        # 加载模型
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=True
        )
        
        self.model = self.model.eval().cuda()
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        # 初始化 TTS
        self.model.init_tts()
        
        print("模型加载完成!")
        
    def mimick_audio(self, audio_path, output_path='output_mimick.wav'):
        """
        语音模仿功能
        
        Args:
            audio_path (str): 输入音频文件路径
            output_path (str): 输出音频文件路径
        """
        if not os.path.exists(audio_path):
            print(f"音频文件不存在: {audio_path}")
            return None
            
        # 加载音频
        audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        # 设置提示词
        mimick_prompt = "请重复每个用户的讲话内容，包括语音风格和内容。"
        
        # 构建消息
        msgs = [{'role': 'user', 'content': [mimick_prompt, audio_input]}]
        
        # 生成响应
        print("正在进行语音模仿...")
        res = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            max_new_tokens=128,
            use_tts_template=True,
            temperature=0.3,
            generate_audio=True,
            output_audio_path=output_path,
        )
        
        print(f"语音模仿完成!")
        print(f"识别文本: {res.text}")
        print(f"输出音频保存到: {output_path}")
        
        return res

def main():
    # 创建演示实例
    demo = AudioMimickDemo()
    
    # 加载模型
    demo.load_model()
    
    # 示例音频文件路径（需要用户提供）
    audio_file = input("请输入要模仿的音频文件路径: ").strip()
    
    if audio_file and os.path.exists(audio_file):
        # 执行语音模仿
        result = demo.mimick_audio(audio_file)
    else:
        print("音频文件路径无效或文件不存在")

if __name__ == "__main__":
    main()
