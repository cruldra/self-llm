#!/bin/bash

# Qwen3-8B vLLM API 测试脚本
# 使用 curl 命令测试 OpenAI 兼容 API

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
DEFAULT_BASE_URL="http://localhost:8000"
DEFAULT_MODEL="Qwen3-8B"

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

# 函数：检查服务器状态
check_server() {
    local base_url="$1"
    
    print_info "检查服务器状态..."
    
    if curl -s -f "$base_url/v1/models" > /dev/null 2>&1; then
        print_success "服务器运行正常"
        return 0
    else
        print_error "无法连接到服务器: $base_url"
        print_info "请确保服务器已启动:"
        print_info "  uv run python start_api_server.py"
        print_info "  或者: bash scripts/start_server.sh"
        return 1
    fi
}

# 函数：测试模型列表接口
test_models() {
    local base_url="$1"
    
    print_info "测试模型列表接口"
    echo "--------------------------------------------------"
    
    local response
    if response=$(curl -s -w "\n%{http_code}" "$base_url/v1/models"); then
        local http_code=$(echo "$response" | tail -n1)
        local body=$(echo "$response" | head -n -1)
        
        if [ "$http_code" = "200" ]; then
            print_success "模型列表获取成功"
            echo "响应内容:"
            echo "$body" | jq '.' 2>/dev/null || echo "$body"
        else
            print_error "HTTP 状态码: $http_code"
            echo "$body"
        fi
    else
        print_error "请求失败"
    fi
    
    echo ""
}

# 函数：测试 Completions API
test_completions() {
    local base_url="$1"
    local model="$2"
    
    print_info "测试 Completions API"
    echo "--------------------------------------------------"
    
    local data='{
        "model": "'$model'",
        "prompt": "我想问你，5的阶乘是多少？<think>\n",
        "max_tokens": 1024,
        "temperature": 0
    }'
    
    print_info "发送请求..."
    echo "请求数据: $data"
    echo ""
    
    local response
    if response=$(curl -s -w "\n%{http_code}" \
        -H "Content-Type: application/json" \
        -d "$data" \
        "$base_url/v1/completions"); then
        
        local http_code=$(echo "$response" | tail -n1)
        local body=$(echo "$response" | head -n -1)
        
        if [ "$http_code" = "200" ]; then
            print_success "Completions API 测试成功"
            echo "响应内容:"
            echo "$body" | jq '.' 2>/dev/null || echo "$body"
        else
            print_error "HTTP 状态码: $http_code"
            echo "$body"
        fi
    else
        print_error "请求失败"
    fi
    
    echo ""
}

# 函数：测试 Chat Completions API
test_chat_completions() {
    local base_url="$1"
    local model="$2"
    
    print_info "测试 Chat Completions API"
    echo "--------------------------------------------------"
    
    local data='{
        "model": "'$model'",
        "messages": [
            {"role": "user", "content": "什么是深度学习？"}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }'
    
    print_info "发送请求..."
    echo "请求数据: $data"
    echo ""
    
    local response
    if response=$(curl -s -w "\n%{http_code}" \
        -H "Content-Type: application/json" \
        -d "$data" \
        "$base_url/v1/chat/completions"); then
        
        local http_code=$(echo "$response" | tail -n1)
        local body=$(echo "$response" | head -n -1)
        
        if [ "$http_code" = "200" ]; then
            print_success "Chat Completions API 测试成功"
            echo "响应内容:"
            echo "$body" | jq '.' 2>/dev/null || echo "$body"
        else
            print_error "HTTP 状态码: $http_code"
            echo "$body"
        fi
    else
        print_error "请求失败"
    fi
    
    echo ""
}

# 函数：测试思考模式
test_thinking_mode() {
    local base_url="$1"
    local model="$2"
    
    print_info "测试思考模式"
    echo "--------------------------------------------------"
    
    local data='{
        "model": "'$model'",
        "messages": [
            {"role": "user", "content": "计算 8 的阶乘，并详细说明计算过程。<think>\n"}
        ],
        "temperature": 0.6,
        "max_tokens": 1024
    }'
    
    print_info "发送思考模式请求..."
    echo "请求数据: $data"
    echo ""
    
    local response
    if response=$(curl -s -w "\n%{http_code}" \
        -H "Content-Type: application/json" \
        -d "$data" \
        "$base_url/v1/chat/completions"); then
        
        local http_code=$(echo "$response" | tail -n1)
        local body=$(echo "$response" | head -n -1)
        
        if [ "$http_code" = "200" ]; then
            print_success "思考模式测试成功"
            echo "响应内容:"
            echo "$body" | jq '.' 2>/dev/null || echo "$body"
        else
            print_error "HTTP 状态码: $http_code"
            echo "$body"
        fi
    else
        print_error "请求失败"
    fi
    
    echo ""
}

# 函数：性能测试
test_performance() {
    local base_url="$1"
    local model="$2"
    
    print_info "性能测试 (简单响应时间)"
    echo "--------------------------------------------------"
    
    local data='{
        "model": "'$model'",
        "messages": [
            {"role": "user", "content": "你好"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }'
    
    print_info "测试响应时间..."
    
    local start_time=$(date +%s.%N)
    local response
    if response=$(curl -s -w "\n%{http_code}" \
        -H "Content-Type: application/json" \
        -d "$data" \
        "$base_url/v1/chat/completions"); then
        
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc -l)
        
        local http_code=$(echo "$response" | tail -n1)
        
        if [ "$http_code" = "200" ]; then
            print_success "性能测试完成"
            printf "响应时间: %.2f 秒\n" "$duration"
        else
            print_error "HTTP 状态码: $http_code"
        fi
    else
        print_error "性能测试失败"
    fi
    
    echo ""
}

# 函数：显示帮助信息
show_help() {
    echo "Qwen3-8B vLLM API 测试脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -u, --url URL                API 服务器地址 (默认: $DEFAULT_BASE_URL)"
    echo "  -m, --model MODEL            模型名称 (默认: $DEFAULT_MODEL)"
    echo "  -t, --test TYPE              测试类型 (all|models|completions|chat|thinking|perf)"
    echo "  --help                       显示此帮助信息"
    echo ""
    echo "测试类型:"
    echo "  all          运行所有测试 (默认)"
    echo "  models       仅测试模型列表接口"
    echo "  completions  仅测试 Completions API"
    echo "  chat         仅测试 Chat Completions API"
    echo "  thinking     仅测试思考模式"
    echo "  perf         仅测试性能"
    echo ""
    echo "示例:"
    echo "  $0                           # 运行所有测试"
    echo "  $0 -t models                 # 仅测试模型列表"
    echo "  $0 -u http://localhost:8080  # 使用不同的服务器地址"
}

# 解析命令行参数
BASE_URL="$DEFAULT_BASE_URL"
MODEL="$DEFAULT_MODEL"
TEST_TYPE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--url)
            BASE_URL="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -t|--test)
            TEST_TYPE="$2"
            shift 2
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

# 主函数
main() {
    echo ""
    print_info "🧪 Qwen3-8B vLLM API 测试"
    echo "=================================================="
    print_info "🌐 服务器地址: $BASE_URL"
    print_info "🏷️  模型名称: $MODEL"
    print_info "🔍 测试类型: $TEST_TYPE"
    echo "=================================================="
    echo ""
    
    # 检查服务器状态
    if ! check_server "$BASE_URL"; then
        exit 1
    fi
    
    echo ""
    
    # 根据测试类型执行相应测试
    case $TEST_TYPE in
        "all")
            test_models "$BASE_URL"
            test_completions "$BASE_URL" "$MODEL"
            test_chat_completions "$BASE_URL" "$MODEL"
            test_thinking_mode "$BASE_URL" "$MODEL"
            test_performance "$BASE_URL" "$MODEL"
            ;;
        "models")
            test_models "$BASE_URL"
            ;;
        "completions")
            test_completions "$BASE_URL" "$MODEL"
            ;;
        "chat")
            test_chat_completions "$BASE_URL" "$MODEL"
            ;;
        "thinking")
            test_thinking_mode "$BASE_URL" "$MODEL"
            ;;
        "perf")
            test_performance "$BASE_URL" "$MODEL"
            ;;
        *)
            print_error "未知的测试类型: $TEST_TYPE"
            show_help
            exit 1
            ;;
    esac
    
    echo "=================================================="
    print_success "🎉 API 测试完成！"
}

# 检查依赖
if ! command -v curl &> /dev/null; then
    print_error "curl 命令未找到，请先安装 curl"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    print_warning "jq 命令未找到，JSON 输出将不会格式化"
fi

if ! command -v bc &> /dev/null; then
    print_warning "bc 命令未找到，性能测试可能无法正常工作"
fi

# 运行主函数
main
