# MiniCPM-o-2.6 多模态语音能力示例项目

本项目演示了 MiniCPM-o-2.6 模型的多模态语音能力，包括语音模仿、音频助手对话、音频理解、语音生成和语音克隆等功能。

## 功能特性

- **语音模仿**: 重复用户的讲话内容，包括语音风格和内容
- **音频助手对话**: 以参考音频中的语音作为 AI 助手进行对话
- **音频角色扮演**: 根据音频提示进行角色扮演对话
- **音频理解**: 包括 ASR、说话人分析、音频总结、场景标记等
- **语音生成**: 根据文本指令生成具有特定风格的语音
- **语音克隆**: 基于参考音频克隆声音并朗读指定文本

## 环境要求

- Python >= 3.12
- CUDA 12.1+
- PyTorch 2.5.1+

## 安装依赖

使用 uv 管理依赖（推荐）：

```bash
# 安装 uv（如果还没有安装）
pip install uv

# 安装项目依赖
uv sync
```

或使用 pip：

```bash
pip install -r requirements.txt
```

## 模型下载

运行模型下载脚本：

```bash
uv run python model_download.py
```

模型将下载到 `./models/MiniCPM-o-2_6/` 目录。

## 使用方法

### 1. 语音模仿

```bash
uv run python audio_mimick.py
```

该脚本会提示您输入要模仿的音频文件路径，然后生成模仿的语音输出。

### 2. 音频助手对话

```bash
uv run python audio_assistant.py
```

支持两种模式：
- 音频助手对话：稳定，适合一般对话
- 音频角色扮演：更自然的对话，但不够稳定

### 3. 音频理解

```bash
uv run python audio_understanding.py
```

支持多种音频理解任务：
- ASR（中文/英文）
- 说话人分析
- 音频总结
- 声音场景标记

### 4. 语音生成

```bash
uv run python voice_generation.py
```

支持两种功能：
- Human Instruction-to-Speech：根据指令生成语音
- 语音克隆：基于参考音频克隆声音

### 5. Jupyter Notebook 演示

启动 Jupyter Notebook：

```bash
uv run jupyter notebook demo_notebook.ipynb
```

Notebook 包含了所有功能的交互式演示，可以直接在浏览器中运行和测试。

## 项目结构

```
MiniCPM-o-2.6-Audio/
├── pyproject.toml              # 项目配置和依赖
├── README.md                   # 项目说明文档
├── model_download.py           # 模型下载脚本
├── audio_mimick.py            # 语音模仿演示
├── audio_assistant.py         # 音频助手对话演示
├── audio_understanding.py     # 音频理解演示
├── voice_generation.py        # 语音生成演示
├── demo_notebook.ipynb        # Jupyter Notebook 综合演示
└── models/                    # 模型文件目录
    └── MiniCPM-o-2_6/        # 下载的模型文件
```

## 使用示例

### 语音模仿示例

```python
from audio_mimick import AudioMimickDemo

demo = AudioMimickDemo()
demo.load_model()
result = demo.mimick_audio('input_audio.wav', 'output_mimick.wav')
```

### 音频助手示例

```python
from audio_assistant import AudioAssistantDemo

demo = AudioAssistantDemo()
demo.load_model()
result = demo.audio_assistant_chat(
    ref_audio_path='ref_voice.wav',
    question_audio_path='question.wav',
    language='zh'
)
```

### 语音生成示例

```python
from voice_generation import VoiceGenerationDemo

demo = VoiceGenerationDemo()
demo.load_model()

# 语音生成
instruction = '一位温柔的女性用甜美的声音轻声说道："今天天气真好。"'
result = demo.generate_voice(instruction)

# 语音克隆
result = demo.voice_cloning(
    ref_audio_path='reference.wav',
    text_to_speak='要克隆的文本内容'
)
```

## 注意事项

1. **GPU 要求**: 模型需要 GPU 支持，建议使用 CUDA 12.1+ 环境
2. **内存要求**: 模型较大，建议至少 16GB 显存
3. **音频格式**: 支持常见音频格式（wav, mp3 等），内部会转换为 16kHz 单声道
4. **文件路径**: 请确保音频文件路径正确且文件存在

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   - 减少 batch size 或使用更小的模型
   - 确保没有其他程序占用 GPU 内存

2. **音频文件加载失败**
   - 检查文件路径是否正确
   - 确保音频文件格式受支持
   - 尝试转换音频文件为 wav 格式

3. **依赖安装问题**
   - 确保 Python 版本 >= 3.12
   - 使用 uv 管理依赖可以避免版本冲突

## 参考资料

- [MiniCPM-o 官方文档](https://github.com/OpenBMB/MiniCPM-o)
- [ModelScope 模型库](https://modelscope.cn/models/OpenBMB/MiniCPM-o-2_6)
- [原始教程文档](../../models/MiniCPM-o/03-MiniCPM-o-2.6%20多模态语音能力.md)

## 许可证

本项目遵循 MIT 许可证。
