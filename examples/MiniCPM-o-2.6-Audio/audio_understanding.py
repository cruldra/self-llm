"""
MiniCPM-o-2.6 音频理解功能演示
"""
import torch
from modelscope import AutoModel, AutoTokenizer
import librosa
import os

class AudioUnderstandingDemo:
    def __init__(self, model_path='./models/MiniCPM-o-2_6'):
        """
        初始化音频理解演示
        
        Args:
            model_path (str): 模型路径
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # 预定义的任务提示词
        self.task_prompts = {
            'asr_zh': '请仔细听这段音频片段，并将其内容逐字记录。',
            'asr_en': 'Please listen to the audio snippet carefully and transcribe the content.',
            'speaker_analysis': 'Based on the speaker\'s content, speculate on their gender, condition, age range, and health status.',
            'audio_summary_zh': '总结音频的主要内容。',
            'audio_summary_en': 'Summarize the main content of the audio.',
            'scene_tagging_zh': '使用一个关键词表达音频内容或相关场景。',
            'scene_tagging_en': 'Utilize one keyword to convey the audio\'s content or the associated scene.'
        }
        
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
        
    def understand_audio(self, audio_path, task='audio_summary_en', 
                        output_path='understanding_result.wav', custom_prompt=None):
        """
        音频理解功能
        
        Args:
            audio_path (str): 输入音频文件路径
            task (str): 任务类型，可选值见 self.task_prompts
            output_path (str): 输出音频文件路径
            custom_prompt (str): 自定义提示词，如果提供则忽略task参数
        """
        if not os.path.exists(audio_path):
            print(f"音频文件不存在: {audio_path}")
            return None
            
        # 加载音频
        audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        # 选择提示词
        if custom_prompt:
            task_prompt = custom_prompt
        elif task in self.task_prompts:
            task_prompt = self.task_prompts[task]
        else:
            print(f"未知任务类型: {task}")
            print(f"可用任务类型: {list(self.task_prompts.keys())}")
            return None
            
        # 构建消息
        msgs = [{'role': 'user', 'content': [task_prompt, audio_input]}]
        
        # 生成响应
        print(f"正在执行音频理解任务: {task}")
        print(f"使用提示词: {task_prompt}")
        
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
        
        print(f"音频理解完成!")
        print(f"理解结果: {res.text}")
        print(f"输出音频保存到: {output_path}")
        
        return res
        
    def show_available_tasks(self):
        """显示可用的任务类型"""
        print("可用的音频理解任务:")
        for task, prompt in self.task_prompts.items():
            print(f"  {task}: {prompt}")

def main():
    # 创建演示实例
    demo = AudioUnderstandingDemo()
    
    # 加载模型
    demo.load_model()
    
    # 显示可用任务
    demo.show_available_tasks()
    
    # 获取用户输入
    audio_file = input("\n请输入要理解的音频文件路径: ").strip()
    
    if not audio_file or not os.path.exists(audio_file):
        print("音频文件路径无效或文件不存在")
        return
        
    print("\n选择任务类型:")
    print("1. 使用预定义任务")
    print("2. 使用自定义提示词")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        task = input("请输入任务类型 (如 asr_zh, audio_summary_en 等): ").strip()
        if task:
            demo.understand_audio(audio_file, task=task)
        else:
            print("任务类型不能为空")
            
    elif choice == "2":
        custom_prompt = input("请输入自定义提示词: ").strip()
        if custom_prompt:
            demo.understand_audio(audio_file, custom_prompt=custom_prompt)
        else:
            print("自定义提示词不能为空")
    else:
        print("无效选择")

if __name__ == "__main__":
    main()
