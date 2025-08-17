"""
MiniCPM-o-2.6 音频助手功能演示
"""
import torch
from modelscope import AutoModel, AutoTokenizer
import librosa
import os

class AudioAssistantDemo:
    def __init__(self, model_path='./models/MiniCPM-o-2_6'):
        """
        初始化音频助手演示
        
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
        
    def audio_assistant_chat(self, ref_audio_path, question_audio_path, 
                           output_path='assistant_response.wav', language='zh'):
        """
        音频助手对话
        
        Args:
            ref_audio_path (str): 参考音频文件路径（用于设置助手声音）
            question_audio_path (str): 问题音频文件路径
            output_path (str): 输出音频文件路径
            language (str): 语言设置 ('zh' 或 'en')
        """
        if not os.path.exists(ref_audio_path):
            print(f"参考音频文件不存在: {ref_audio_path}")
            return None
            
        if not os.path.exists(question_audio_path):
            print(f"问题音频文件不存在: {question_audio_path}")
            return None
            
        # 加载参考音频
        ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        
        # 获取系统提示
        sys_prompt = self.model.get_sys_prompt(
            ref_audio=ref_audio, 
            mode='audio_assistant', 
            language=language
        )
        
        # 加载问题音频
        question_audio, _ = librosa.load(question_audio_path, sr=16000, mono=True)
        user_question = {'role': 'user', 'content': [question_audio]}
        
        # 构建消息
        msgs = [sys_prompt, user_question]
        
        # 生成响应
        print("正在生成音频助手响应...")
        res = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            max_new_tokens=128,
            use_tts_template=True,
            generate_audio=True,
            temperature=0.3,
            output_audio_path=output_path,
        )
        
        print(f"音频助手响应完成!")
        print(f"响应文本: {res.text}")
        print(f"输出音频保存到: {output_path}")
        
        return res
        
    def audio_roleplay_chat(self, ref_audio_path, question_audio_path, 
                          output_path='roleplay_response.wav', language='en'):
        """
        音频角色扮演对话
        
        Args:
            ref_audio_path (str): 参考音频文件路径（用于设置角色声音）
            question_audio_path (str): 问题音频文件路径
            output_path (str): 输出音频文件路径
            language (str): 语言设置 ('zh' 或 'en')
        """
        if not os.path.exists(ref_audio_path):
            print(f"参考音频文件不存在: {ref_audio_path}")
            return None
            
        if not os.path.exists(question_audio_path):
            print(f"问题音频文件不存在: {question_audio_path}")
            return None
            
        # 加载参考音频
        ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        
        # 获取系统提示
        sys_prompt = self.model.get_sys_prompt(
            ref_audio=ref_audio, 
            mode='audio_roleplay', 
            language=language
        )
        
        # 加载问题音频
        question_audio, _ = librosa.load(question_audio_path, sr=16000, mono=True)
        user_question = {'role': 'user', 'content': [question_audio]}
        
        # 构建消息
        msgs = [sys_prompt, user_question]
        
        # 生成响应
        print("正在生成角色扮演响应...")
        res = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            max_new_tokens=128,
            use_tts_template=True,
            generate_audio=True,
            temperature=0.3,
            output_audio_path=output_path,
        )
        
        print(f"角色扮演响应完成!")
        print(f"响应文本: {res.text}")
        print(f"输出音频保存到: {output_path}")
        
        return res

def main():
    # 创建演示实例
    demo = AudioAssistantDemo()
    
    # 加载模型
    demo.load_model()
    
    print("选择功能:")
    print("1. 音频助手对话")
    print("2. 音频角色扮演")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        ref_audio = input("请输入参考音频文件路径: ").strip()
        question_audio = input("请输入问题音频文件路径: ").strip()
        language = input("请输入语言设置 (zh/en, 默认zh): ").strip() or 'zh'
        
        if ref_audio and question_audio:
            demo.audio_assistant_chat(ref_audio, question_audio, language=language)
        else:
            print("音频文件路径不能为空")
            
    elif choice == "2":
        ref_audio = input("请输入参考音频文件路径: ").strip()
        question_audio = input("请输入问题音频文件路径: ").strip()
        language = input("请输入语言设置 (zh/en, 默认en): ").strip() or 'en'
        
        if ref_audio and question_audio:
            demo.audio_roleplay_chat(ref_audio, question_audio, language=language)
        else:
            print("音频文件路径不能为空")
    else:
        print("无效选择")

if __name__ == "__main__":
    main()
