#!/bin/bash

# Qwen3-8B vLLM API 服务器启动脚本
# 使用 vLLM 启动兼容 OpenAI API 的服务器

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
DEFAULT_MODEL_PATH=""
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"
DEFAULT_MODEL_NAME="Qwen3-8B"
DEFAULT_MAX_LEN="8192"
DEFAULT_GPU_UTIL="0.9"
DEFAULT_PARALLEL_SIZE="1"

# 函数：打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 函数：检测模型路径
detect_model_path() {
    # 优先检查 AutoDL 环境
    if [ -d "/root/autodl-tmp/Qwen/Qwen3-8B" ]; then
        echo "/root/autodl-tmp/Qwen/Qwen3-8B"
        return 0
    fi
    
    # 检查本地 models 目录
    if [ -d "./models/Qwen/Qwen3-8B" ]; then
        echo "./models/Qwen/Qwen3-8B"
        return 0
    fi
    
    # 检查相对路径
    if [ -d "../models/Qwen/Qwen3-8B" ]; then
        echo "../models/Qwen/Qwen3-8B"
        return 0
    fi
    
    return 1
}

# 函数：检查模型文件
check_model_files() {
    local model_path="$1"
    
    if [ ! -d "$model_path" ]; then
        print_error "模型目录不存在: $model_path"
        return 1
    fi
    
    # 检查关键文件
    local required_files=("config.json" "tokenizer.json" "tokenizer_config.json")
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$model_path/$file" ]; then
            print_error "缺少模型文件: $file"
            return 1
        fi
    done
    
    return 0
}

# 函数：显示帮助信息
show_help() {
    echo "Qwen3-8B vLLM API 服务器启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --model-path PATH        模型路径 (默认自动检测)"
    echo "  -h, --host HOST              服务器主机地址 (默认: $DEFAULT_HOST)"
    echo "  -p, --port PORT              服务器端口 (默认: $DEFAULT_PORT)"
    echo "  -n, --model-name NAME        服务模型名称 (默认: $DEFAULT_MODEL_NAME)"
    echo "  -l, --max-len LENGTH         模型最大长度 (默认: $DEFAULT_MAX_LEN)"
    echo "  -g, --gpu-util RATIO         GPU 内存利用率 (默认: $DEFAULT_GPU_UTIL)"
    echo "  -t, --tensor-parallel SIZE   张量并行大小 (默认: $DEFAULT_PARALLEL_SIZE)"
    echo "  --disable-reasoning          禁用推理模式"
    echo "  --help                       显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认配置启动"
    echo "  $0 -p 8080                           # 在端口 8080 启动"
    echo "  $0 -m /path/to/model --disable-reasoning  # 指定模型路径并禁用推理模式"
}

# 解析命令行参数
MODEL_PATH=""
HOST="$DEFAULT_HOST"
PORT="$DEFAULT_PORT"
MODEL_NAME="$DEFAULT_MODEL_NAME"
MAX_LEN="$DEFAULT_MAX_LEN"
GPU_UTIL="$DEFAULT_GPU_UTIL"
PARALLEL_SIZE="$DEFAULT_PARALLEL_SIZE"
ENABLE_REASONING=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -n|--model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        -l|--max-len)
            MAX_LEN="$2"
            shift 2
            ;;
        -g|--gpu-util)
            GPU_UTIL="$2"
            shift 2
            ;;
        -t|--tensor-parallel)
            PARALLEL_SIZE="$2"
            shift 2
            ;;
        --disable-reasoning)
            ENABLE_REASONING=false
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检测模型路径
if [ -z "$MODEL_PATH" ]; then
    print_info "自动检测模型路径..."
    if MODEL_PATH=$(detect_model_path); then
        print_success "找到模型路径: $MODEL_PATH"
    else
        print_error "未找到模型文件"
        print_info "请先下载模型:"
        print_info "  uv run python model_download.py"
        exit 1
    fi
fi

# 检查模型文件
print_info "检查模型文件..."
if ! check_model_files "$MODEL_PATH"; then
    print_error "模型文件检查失败"
    exit 1
fi
print_success "模型文件检查通过"

# 设置环境变量
export VLLM_USE_MODELSCOPE=true

# 构建启动命令
CMD="vllm serve \"$MODEL_PATH\""
CMD="$CMD --host $HOST"
CMD="$CMD --port $PORT"
CMD="$CMD --served-model-name $MODEL_NAME"
CMD="$CMD --max-model-len $MAX_LEN"
CMD="$CMD --gpu-memory-utilization $GPU_UTIL"
CMD="$CMD --tensor-parallel-size $PARALLEL_SIZE"
CMD="$CMD --trust-remote-code"

if [ "$ENABLE_REASONING" = true ]; then
    CMD="$CMD --enable-reasoning --reasoning-parser deepseek_r1"
fi

# 显示启动信息
echo ""
print_info "🚀 启动 vLLM API 服务器"
echo "=================================================="
print_info "📁 模型路径: $MODEL_PATH"
print_info "🌐 服务地址: http://$HOST:$PORT"
print_info "🏷️  模型名称: $MODEL_NAME"
print_info "📏 最大长度: $MAX_LEN"
print_info "🧠 推理模式: $([ "$ENABLE_REASONING" = true ] && echo "启用" || echo "禁用")"
print_info "🎯 GPU 利用率: $GPU_UTIL"
print_info "⚡ 并行大小: $PARALLEL_SIZE"
echo "=================================================="

print_info "执行命令:"
echo "$CMD"
echo ""

print_info "⏳ 正在启动服务器..."
print_info "💡 提示: 服务器启动后，可以通过以下方式测试:"
print_info "   curl http://localhost:$PORT/v1/models"
print_info "   uv run python test_openai_chat_completions.py"
echo ""
print_warning "按 Ctrl+C 停止服务器"
echo "--------------------------------------------------"

# 启动服务器
eval $CMD
