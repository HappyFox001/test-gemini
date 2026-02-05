#!/bin/bash

# 检查 API Key 是否设置
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ 错误: GEMINI_API_KEY 未设置"
    echo ""
    echo "请先设置环境变量:"
    echo "  export GEMINI_API_KEY='your-api-key-here'"
    echo ""
    exit 1
fi

echo "✅ API Key 已设置"
echo ""

# 运行测试
python3 test_gemini_models.py
