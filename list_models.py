#!/usr/bin/env python3
"""
列出所有可用的 Gemini 模型
"""

from google import genai

API_KEY = "AIzaSyCwZrY_wdEnT_g8mWsvTc3kjFZg7h1HsVY"

try:
    client = genai.Client(api_key=API_KEY)

    print("\n" + "=" * 80)
    print("📋 可用的 Gemini 模型列表")
    print("=" * 80 + "\n")

    models = client.models.list()

    gemini_models = []

    for model in models:
        if hasattr(model, 'name'):
            model_name = model.name
            # 只显示 gemini 模型
            if 'gemini' in model_name.lower():
                gemini_models.append(model)
                print(f"✓ {model_name}")

                # 显示支持的方法
                if hasattr(model, 'supported_generation_methods'):
                    methods = model.supported_generation_methods
                    if methods:
                        print(f"  支持的方法: {', '.join(methods)}")

                print()

    print("=" * 80)
    print(f"\n找到 {len(gemini_models)} 个 Gemini 模型\n")

except Exception as e:
    print(f"\n❌ 错误: {str(e)}\n")
    print("请确保:")
    print("1. 已设置有效的 GOOGLE_API_KEY")
    print("2. API Key 有权限访问 Gemini API")
    print()
