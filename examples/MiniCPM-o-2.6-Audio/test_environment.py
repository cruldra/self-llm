"""
MiniCPM-o-2.6 环境测试脚本
"""
import sys
import importlib
import torch

def test_python_version():
    """测试 Python 版本"""
    print("=" * 50)
    print("Python 版本检查")
    print("=" * 50)
    
    version = sys.version_info
    print(f"当前 Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 12:
        print("✓ Python 版本满足要求 (>= 3.12)")
        return True
    else:
        print("✗ Python 版本不满足要求，需要 >= 3.12")
        return False

def test_cuda_availability():
    """测试 CUDA 可用性"""
    print("\n" + "=" * 50)
    print("CUDA 环境检查")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print("✓ CUDA 可用")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
        return True
    else:
        print("✗ CUDA 不可用")
        print("注意: MiniCPM-o-2.6 需要 GPU 支持")
        return False

def test_required_packages():
    """测试必需的包"""
    print("\n" + "=" * 50)
    print("依赖包检查")
    print("=" * 50)
    
    required_packages = [
        'torch',
        'transformers',
        'modelscope',
        'librosa',
        'soundfile',
        'accelerate',
        'timm',
        'vector_quantize_pytorch',
        'vocos'
    ]
    
    all_available = True
    
    for package in required_packages:
        try:
            # 特殊处理一些包名
            if package == 'vector_quantize_pytorch':
                import_name = 'vector_quantize_pytorch'
            else:
                import_name = package
                
            module = importlib.import_module(import_name)
            
            # 获取版本信息
            version = getattr(module, '__version__', 'Unknown')
            print(f"✓ {package}: {version}")
            
        except ImportError:
            print(f"✗ {package}: 未安装")
            all_available = False
        except Exception as e:
            print(f"? {package}: 检查时出错 ({e})")
            all_available = False
    
    return all_available

def test_model_path():
    """测试模型路径"""
    print("\n" + "=" * 50)
    print("模型文件检查")
    print("=" * 50)
    
    import os
    model_path = './models/MiniCPM-o-2_6'
    
    if os.path.exists(model_path):
        print(f"✓ 模型目录存在: {model_path}")
        
        # 检查关键文件
        key_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        files_exist = []
        
        for file in key_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                files_exist.append(file)
                print(f"  ✓ {file}")
            else:
                print(f"  ? {file} (可能不存在)")
        
        if len(files_exist) > 0:
            print("✓ 模型文件部分可用")
            return True
        else:
            print("? 模型文件可能不完整")
            return False
    else:
        print(f"✗ 模型目录不存在: {model_path}")
        print("请运行 'uv run python model_download.py' 下载模型")
        return False

def test_audio_processing():
    """测试音频处理功能"""
    print("\n" + "=" * 50)
    print("音频处理功能检查")
    print("=" * 50)
    
    try:
        import librosa
        import soundfile as sf
        import numpy as np
        
        # 创建测试音频数据
        sample_rate = 16000
        duration = 1.0  # 1秒
        frequency = 440  # A4音符
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # 测试保存和加载
        test_file = 'test_audio.wav'
        sf.write(test_file, audio_data, sample_rate)
        
        loaded_audio, loaded_sr = librosa.load(test_file, sr=16000, mono=True)
        
        # 清理测试文件
        import os
        os.remove(test_file)
        
        print("✓ 音频处理功能正常")
        print(f"  - 生成音频长度: {len(audio_data)} 样本")
        print(f"  - 加载音频长度: {len(loaded_audio)} 样本")
        print(f"  - 采样率: {loaded_sr} Hz")
        
        return True
        
    except Exception as e:
        print(f"✗ 音频处理功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("MiniCPM-o-2.6 环境测试")
    print("测试开始...\n")
    
    results = []
    
    # 运行各项测试
    results.append(("Python 版本", test_python_version()))
    results.append(("CUDA 环境", test_cuda_availability()))
    results.append(("依赖包", test_required_packages()))
    results.append(("模型文件", test_model_path()))
    results.append(("音频处理", test_audio_processing()))
    
    # 总结结果
    print("\n" + "=" * 50)
    print("测试结果总结")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！环境配置正确，可以开始使用 MiniCPM-o-2.6")
    else:
        print(f"\n⚠️  有 {total - passed} 项测试失败，请检查环境配置")
        
        if not results[1][1]:  # CUDA 测试失败
            print("提示: 如果没有 GPU，某些功能可能无法正常工作")
        if not results[2][1]:  # 依赖包测试失败
            print("提示: 请运行 'uv sync' 安装依赖包")
        if not results[3][1]:  # 模型文件测试失败
            print("提示: 请运行 'uv run python model_download.py' 下载模型")

if __name__ == "__main__":
    main()
