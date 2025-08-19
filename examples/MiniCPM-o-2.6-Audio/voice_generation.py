"""
MiniCPM-o-2.6 语音生成功能演示
"""
import torch
from modelscope import AutoModel, AutoTokenizer
import librosa
import os

class VoiceGenerationDemo:
    def __init__(self, model_path='./models/MiniCPM-o-2_6'):
        """
        初始化语音生成演示
        
        Args:
            model_path (str): 模型路径
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # 预定义的语音生成示例
        self.generation_examples = {
            'zh_news_male': '在新闻中，一个年轻男性兴致勃勃地说："祝福亲爱的祖国母亲美丽富强！"他用低音调和低音量，慢慢地说出了这句话。',
            'en_surprised_male': 'Delighting in a surprised tone, an adult male with low pitch and low volume comments:"One even gave my little dog a biscuit" This dialogue takes place at a leisurely pace, delivering a sense of excitement and surprise in the context.',
            'zh_gentle_female': '一位温柔的女性用甜美的声音轻声说道："今天天气真好，适合出去散步。"她的语调平缓，充满了温暖。',
            'en_excited_child': 'A cheerful child excitedly exclaims: "I got a new toy today!" The voice is high-pitched and full of joy, speaking at a fast pace.',
            'zh_serious_announcer': '播音员用严肃而庄重的语调宣布："现在播报重要新闻。"声音低沉有力，语速适中。'
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
        
    def generate_voice(self, instruction_prompt, output_path='generated_voice.wav'):
        """
        根据指令生成语音
        
        Args:
            instruction_prompt (str): 语音生成指令
            output_path (str): 输出音频文件路径
        """
        # 构建消息
        msgs = [{'role': 'user', 'content': [instruction_prompt]}]
        
        # 生成响应
        print("正在生成语音...")
        print(f"指令: {instruction_prompt}")
        
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
        
        print(f"语音生成完成!")
        print(f"生成文本: {res.text}")
        print(f"输出音频保存到: {output_path}")
        
        return res
        
    def voice_cloning(self, ref_audio_path, text_to_speak, 
                     output_path='cloned_voice.wav', language='zh'):
        """
        语音克隆功能
        
        Args:
            ref_audio_path (str): 参考音频文件路径
            text_to_speak (str): 要说的文本内容
            output_path (str): 输出音频文件路径
            language (str): 语言设置 ('zh' 或 'en')
        """
        if not os.path.exists(ref_audio_path):
            print(f"参考音频文件不存在: {ref_audio_path}")
            return None
            
        # 加载参考音频
        ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        
        # 获取系统提示
        sys_prompt = self.model.get_sys_prompt(
            ref_audio=ref_audio, 
            mode='voice_cloning', 
            language=language
        )
        
        # 构建用户消息
        text_prompt = "Please read the text below."
        user_question = {'role': 'user', 'content': [text_prompt, text_to_speak]}
        
        # 构建消息
        msgs = [sys_prompt, user_question]
        
        # 生成响应
        print("正在进行语音克隆...")
        print(f"要克隆的文本: {text_to_speak}")
        
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
        
        print(f"语音克隆完成!")
        print(f"克隆文本: {res.text}")
        print(f"输出音频保存到: {output_path}")
        
        return res
        
    def show_generation_examples(self):
        """显示语音生成示例"""
        print("语音生成指令示例:")
        for key, example in self.generation_examples.items():
            print(f"  {key}: {example}")

def main():
    # 创建演示实例
    demo = VoiceGenerationDemo()
    
    # 加载模型
    demo.load_model()
    
    print("选择功能:")
    print("1. 语音生成 (Human Instruction-to-Speech)")
    print("2. 语音克隆 (Voice Cloning)")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        print("\n语音生成功能")
        demo.show_generation_examples()
        
        print("\n选择输入方式:")
        print("1. 使用预定义示例")
        print("2. 自定义指令")
        
        input_choice = input("请输入选择 (1 或 2): ").strip()
        
        if input_choice == "1":
            example_key = input("请输入示例键名: ").strip()
            if example_key in demo.generation_examples:
                instruction = demo.generation_examples[example_key]
                demo.generate_voice(instruction)
            else:
                print("无效的示例键名")
        elif input_choice == "2":
            instruction = input("请输入语音生成指令: ").strip()
            if instruction:
                demo.generate_voice(instruction)
            else:
                print("指令不能为空")
        else:
            print("无效选择")
            
    elif choice == "2":
        print("\n语音克隆功能")
        ref_audio = input("请输入参考音频文件路径: ").strip()
        text_to_speak = input("请输入要说的文本: ").strip()
        language = input("请输入语言设置 (zh/en, 默认zh): ").strip() or 'zh'
        
        if ref_audio and text_to_speak:
            demo.voice_cloning(ref_audio, text_to_speak, language=language)
        else:
            print("参考音频路径和文本内容不能为空")
    else:
        print("无效选择")

if __name__ == "__main__":
    main()
