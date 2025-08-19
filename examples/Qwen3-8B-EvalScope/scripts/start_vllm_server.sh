#!/bin/bash

# Qwen3-8B vLLM 服务启动脚本
# 
# 使用方法:
#   bash scripts/start_vllm_server.sh
#   bash scripts/start_vllm_server.sh --model-path /path/to/model
#   bash scripts/start_vllm_server.sh --port 8001

set -e

# 默认配置
MODEL_PATH="./models/Qwen/Qwen3-8B"
HOST="0.0.0.0"
PORT="8000"
SERVED_MODEL_NAME="Qwen3-8B"
MAX_MODEL_LEN="8192"
GPU_MEMORY_UTILIZATION="0.9"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --served-model-name)
            SERVED_MODEL_NAME="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        -h|--help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --model-path PATH              模型路径 (默认: ./models/Qwen/Qwen3-8B)"
            echo "  --host HOST                    服务器主机 (默认: 0.0.0.0)"
            echo "  --port PORT                    服务器端口 (默认: 8000)"
            echo "  --served-model-name NAME       服务模型名称 (默认: Qwen3-8B)"
            echo "  --max-model-len LENGTH         最大模型长度 (默认: 8192)"
            echo "  --gpu-memory-utilization RATIO GPU 内存利用率 (默认: 0.9)"
            echo "  -h, --help                     显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查模型是否存在
check_model() {
    print_info "检查模型路径: $MODEL_PATH"
    
    if [ ! -d "$MODEL_PATH" ]; then
        print_error "模型目录不存在: $MODEL_PATH"
        print_info "请先下载模型: uv run python model_download.py"
        exit 1
    fi
    
    # 检查关键文件
    required_files=("config.json" "tokenizer.json" "tokenizer_config.json")
    for file in "${required_files[@]}"; do
        if [ ! -f "$MODEL_PATH/$file" ]; then
            print_error "缺少模型文件: $MODEL_PATH/$file"
            exit 1
        fi
    done
    
    print_success "模型文件检查通过"
}

# 检查端口是否被占用
check_port() {
    print_info "检查端口 $PORT 是否可用..."
    
    if command -v lsof >/dev/null 2>&1; then
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null; then
            print_warning "端口 $PORT 已被占用"
            print_info "尝试检查服务是否已运行..."
            
            # 检查健康状态
            if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
                print_success "vLLM 服务已在运行"
                print_info "API 地址: http://localhost:$PORT/v1"
                exit 0
            else
                print_error "端口被其他服务占用，请更换端口或停止占用服务"
                exit 1
            fi
        fi
    else
        print_warning "无法检查端口状态 (lsof 命令不可用)"
    fi
    
    print_success "端口 $PORT 可用"
}

# 检查 GPU
check_gpu() {
    print_info "检查 GPU 状态..."
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
        print_info "检测到 $gpu_count 个 GPU"
        
        # 显示 GPU 信息
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits | while read line; do
            print_info "GPU: $line"
        done
    else
        print_warning "未检测到 NVIDIA GPU 或 nvidia-smi 命令不可用"
        print_warning "将使用 CPU 模式运行 (性能较低)"
    fi
}

# 启动 vLLM 服务
start_vllm() {
    print_info "启动 vLLM 服务..."
    print_info "模型路径: $MODEL_PATH"
    print_info "服务地址: http://$HOST:$PORT"
    print_info "模型名称: $SERVED_MODEL_NAME"
    print_info "最大长度: $MAX_MODEL_LEN"
    print_info "GPU 内存利用率: $GPU_MEMORY_UTILIZATION"
    
    echo "=================================="
    
    # 设置环境变量
    export VLLM_USE_MODELSCOPE=true
    
    # 构建启动命令
    cmd="vllm serve \"$MODEL_PATH\" \
        --host $HOST \
        --port $PORT \
        --served-model-name \"$SERVED_MODEL_NAME\" \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --enable-reasoning \
        --reasoning-parser deepseek_r1"
    
    print_info "执行命令: $cmd"
    echo "=================================="
    
    # 启动服务
    eval $cmd
}

# 主函数
main() {
    print_info "🚀 启动 Qwen3-8B vLLM 服务"
    echo "=================================="
    
    # 检查模型
    check_model
    
    # 检查端口
    check_port
    
    # 检查 GPU
    check_gpu
    
    # 启动服务
    start_vllm
}

# 信号处理
trap 'print_warning "收到中断信号，正在停止服务..."; exit 130' INT TERM

# 运行主函数
main "$@"
