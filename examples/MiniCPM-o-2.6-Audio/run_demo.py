"""
MiniCPM-o-2.6 多模态语音能力演示启动脚本
"""
import os
import sys

def show_menu():
    """显示功能菜单"""
    print("=" * 50)
    print("MiniCPM-o-2.6 多模态语音能力演示")
    print("=" * 50)
    print("1. 下载模型")
    print("2. 语音模仿演示")
    print("3. 音频助手对话演示")
    print("4. 音频理解演示")
    print("5. 语音生成演示")
    print("6. 启动 Jupyter Notebook")
    print("0. 退出")
    print("=" * 50)

def check_model_exists():
    """检查模型是否存在"""
    model_path = './models/MiniCPM-o-2_6'
    return os.path.exists(model_path) and os.path.isdir(model_path)

def download_model():
    """下载模型"""
    print("开始下载 MiniCPM-o-2.6 模型...")
    os.system("uv run python model_download.py")

def run_audio_mimick():
    """运行语音模仿演示"""
    if not check_model_exists():
        print("模型不存在，请先下载模型（选项1）")
        return
    print("启动语音模仿演示...")
    os.system("uv run python audio_mimick.py")

def run_audio_assistant():
    """运行音频助手演示"""
    if not check_model_exists():
        print("模型不存在，请先下载模型（选项1）")
        return
    print("启动音频助手演示...")
    os.system("uv run python audio_assistant.py")

def run_audio_understanding():
    """运行音频理解演示"""
    if not check_model_exists():
        print("模型不存在，请先下载模型（选项1）")
        return
    print("启动音频理解演示...")
    os.system("uv run python audio_understanding.py")

def run_voice_generation():
    """运行语音生成演示"""
    if not check_model_exists():
        print("模型不存在，请先下载模型（选项1）")
        return
    print("启动语音生成演示...")
    os.system("uv run python voice_generation.py")

def run_jupyter_notebook():
    """启动 Jupyter Notebook"""
    if not check_model_exists():
        print("模型不存在，请先下载模型（选项1）")
        return
    print("启动 Jupyter Notebook...")
    os.system("uv run jupyter notebook demo_notebook.ipynb")

def main():
    """主函数"""
    while True:
        show_menu()
        
        try:
            choice = input("请选择功能 (0-6): ").strip()
            
            if choice == "0":
                print("退出程序")
                sys.exit(0)
            elif choice == "1":
                download_model()
            elif choice == "2":
                run_audio_mimick()
            elif choice == "3":
                run_audio_assistant()
            elif choice == "4":
                run_audio_understanding()
            elif choice == "5":
                run_voice_generation()
            elif choice == "6":
                run_jupyter_notebook()
            else:
                print("无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            sys.exit(0)
        except Exception as e:
            print(f"发生错误: {e}")
            
        input("\n按回车键继续...")

if __name__ == "__main__":
    main()
