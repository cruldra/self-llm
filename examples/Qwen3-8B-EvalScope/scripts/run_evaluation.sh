#!/bin/bash

# Qwen3-8B EvalScope 评测运行脚本
#
# 使用方法:
#   bash scripts/run_evaluation.sh                    # 默认 IQuiz 评测
#   bash scripts/run_evaluation.sh --preset stable   # 使用稳定配置
#   bash scripts/run_evaluation.sh --datasets iquiz,cmmlu  # 多数据集评测

set -e

# 默认配置
MODEL_NAME="Qwen3-8B"
API_URL="http://localhost:8000/v1"
API_KEY="EMPTY"
DATASETS="iquiz"
WORK_DIR="outputs/Qwen3-8B"
EVAL_BATCH_SIZE="16"
TEMPERATURE="0.7"
MAX_TOKENS="4096"
PRESET=""
METHOD="api"  # api 或 cli

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --eval-batch-size)
            EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        -h|--help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --model-name NAME              模型名称 (默认: Qwen3-8B)"
            echo "  --api-url URL                  API 地址 (默认: http://localhost:8000/v1)"
            echo "  --api-key KEY                  API 密钥 (默认: EMPTY)"
            echo "  --datasets DATASETS            数据集列表，逗号分隔 (默认: iquiz)"
            echo "  --work-dir DIR                 工作目录 (默认: outputs/Qwen3-8B)"
            echo "  --eval-batch-size SIZE         评测批次大小 (默认: 16)"
            echo "  --temperature TEMP             温度参数 (默认: 0.7)"
            echo "  --max-tokens TOKENS            最大 tokens (默认: 4096)"
            echo "  --preset PRESET                预设配置 (stable, creative, comprehensive)"
            echo "  --method METHOD                评测方法 (api, cli) (默认: api)"
            echo "  -h, --help                     显示帮助信息"
            echo ""
            echo "预设配置:"
            echo "  stable                         稳定配置 (低温度)"
            echo "  creative                       创意配置 (高温度)"
            echo "  comprehensive                  综合评测 (多数据集)"
            echo ""
            echo "支持的数据集:"
            echo "  iquiz                          智商情商测试"
            echo "  mmlu                           大规模多任务语言理解"
            echo "  cmmlu                          中文大规模多任务语言理解"
            echo "  ceval                          中文评测基准"
            echo "  gsm8k                          小学数学应用题"
            echo "  math                           高中数学竞赛题"
            echo "  humaneval                      代码生成评测"
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

# 应用预设配置
apply_preset() {
    case $PRESET in
        stable)
            TEMPERATURE="0.1"
            WORK_DIR="outputs/Qwen3-8B/iquiz_stable"
            print_info "应用稳定配置预设"
            ;;
        creative)
            TEMPERATURE="1.0"
            WORK_DIR="outputs/Qwen3-8B/iquiz_creative"
            print_info "应用创意配置预设"
            ;;
        comprehensive)
            DATASETS="iquiz,cmmlu,ceval"
            EVAL_BATCH_SIZE="8"
            WORK_DIR="outputs/Qwen3-8B/comprehensive"
            print_info "应用综合评测预设"
            ;;
        *)
            if [ -n "$PRESET" ]; then
                print_error "未知的预设配置: $PRESET"
                exit 1
            fi
            ;;
    esac
}

# 检查 API 服务
check_api_service() {
    print_info "检查 API 服务状态..."
    
    health_url=$(echo "$API_URL" | sed 's|/v1.*|/health|')
    
    if curl -s "$health_url" >/dev/null 2>&1; then
        print_success "API 服务正常: $health_url"
    else
        print_error "API 服务不可用: $health_url"
        print_info "请先启动 vLLM 服务:"
        print_info "  bash scripts/start_vllm_server.sh"
        print_info "  或者: uv run python start_vllm_server.py"
        exit 1
    fi
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."
    
    # 检查 Python
    if ! command -v python >/dev/null 2>&1; then
        print_error "Python 未安装"
        exit 1
    fi
    
    # 检查 evalscope (仅在 CLI 模式下)
    if [ "$METHOD" = "cli" ]; then
        if ! command -v evalscope >/dev/null 2>&1; then
            print_error "evalscope 命令不可用"
            print_info "请安装 EvalScope: pip install evalscope"
            exit 1
        fi
        print_success "evalscope 命令可用"
    fi
    
    print_success "依赖检查通过"
}

# 运行 API 评测
run_api_evaluation() {
    print_info "使用 Python API 运行评测..."
    
    # 构建参数
    args=(
        "--model-name" "$MODEL_NAME"
        "--api-url" "$API_URL"
        "--api-key" "$API_KEY"
        "--work-dir" "$WORK_DIR"
        "--temperature" "$TEMPERATURE"
        "--max-tokens" "$MAX_TOKENS"
    )
    
    print_info "执行命令: uv run python eval_api.py ${args[*]}"
    
    # 运行评测
    uv run python eval_api.py "${args[@]}"
}

# 运行 CLI 评测
run_cli_evaluation() {
    print_info "使用 EvalScope CLI 运行评测..."
    
    # 将数据集字符串转换为数组
    IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
    
    for dataset in "${DATASET_ARRAY[@]}"; do
        dataset=$(echo "$dataset" | xargs)  # 去除空格
        
        print_info "评测数据集: $dataset"
        
        # 构建命令
        cmd=(
            "evalscope" "eval"
            "--model" "$MODEL_NAME"
            "--api-url" "$API_URL"
            "--api-key" "$API_KEY"
            "--eval-type" "service"
            "--eval-batch-size" "$EVAL_BATCH_SIZE"
            "--datasets" "$dataset"
            "--work-dir" "$WORK_DIR/$dataset"
        )
        
        print_info "执行命令: ${cmd[*]}"
        
        # 运行评测
        "${cmd[@]}"
        
        print_success "数据集 $dataset 评测完成"
    done
}

# 显示结果
show_results() {
    print_info "评测结果位置: $WORK_DIR"
    
    if [ -d "$WORK_DIR" ]; then
        print_info "结果目录内容:"
        ls -la "$WORK_DIR" | head -10
        
        # 查找最新结果
        latest_dir=$(find "$WORK_DIR" -type d -name "20*" | sort | tail -1)
        if [ -n "$latest_dir" ]; then
            print_info "最新结果目录: $latest_dir"
            
            # 显示摘要文件
            summary_files=$(find "$latest_dir" -name "*summary*" -o -name "*result*" | head -5)
            if [ -n "$summary_files" ]; then
                print_info "结果文件:"
                echo "$summary_files"
            fi
        fi
    fi
}

# 主函数
main() {
    print_info "🧠 开始 Qwen3-8B EvalScope 评测"
    echo "=================================="
    
    # 应用预设配置
    if [ -n "$PRESET" ]; then
        apply_preset
    fi
    
    # 显示配置
    print_info "评测配置:"
    print_info "  模型: $MODEL_NAME"
    print_info "  API: $API_URL"
    print_info "  数据集: $DATASETS"
    print_info "  工作目录: $WORK_DIR"
    print_info "  批次大小: $EVAL_BATCH_SIZE"
    print_info "  温度: $TEMPERATURE"
    print_info "  最大 tokens: $MAX_TOKENS"
    print_info "  方法: $METHOD"
    echo "=================================="
    
    # 检查依赖
    check_dependencies
    
    # 检查 API 服务
    check_api_service
    
    # 创建工作目录
    mkdir -p "$WORK_DIR"
    
    # 记录开始时间
    start_time=$(date +%s)
    
    # 运行评测
    case $METHOD in
        api)
            run_api_evaluation
            ;;
        cli)
            run_cli_evaluation
            ;;
        *)
            print_error "未知的评测方法: $METHOD"
            exit 1
            ;;
    esac
    
    # 计算耗时
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    print_success "评测完成！耗时: ${duration} 秒"
    
    # 显示结果
    show_results
    
    print_success "🎉 评测任务完成！"
}

# 信号处理
trap 'print_warning "收到中断信号，正在停止评测..."; exit 130' INT TERM

# 运行主函数
main "$@"
