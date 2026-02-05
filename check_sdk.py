#!/usr/bin/env python3
"""
检查 Google GenAI SDK 的版本和可用属性
"""

import google.genai
from google.genai import types
import inspect

print("=" * 80)
print("Google GenAI SDK 信息")
print("=" * 80)

# 检查 SDK 版本
if hasattr(google.genai, '__version__'):
    print(f"SDK 版本: {google.genai.__version__}")
else:
    print("SDK 版本: 未知")

print("\n" + "=" * 80)
print("检查 types 模块中的属性")
print("=" * 80)

# 检查 types 模块中是否有 ThinkingLevel
print(f"\n是否有 ThinkingLevel: {hasattr(types, 'ThinkingLevel')}")
if hasattr(types, 'ThinkingLevel'):
    print(f"ThinkingLevel 值: {[attr for attr in dir(types.ThinkingLevel) if not attr.startswith('_')]}")

# 检查 ThinkingConfig
print(f"\n是否有 ThinkingConfig: {hasattr(types, 'ThinkingConfig')}")
if hasattr(types, 'ThinkingConfig'):
    print("\nThinkingConfig 签名:")
    sig = inspect.signature(types.ThinkingConfig)
    print(f"  参数: {list(sig.parameters.keys())}")

    # 尝试查看类的字段
    if hasattr(types.ThinkingConfig, '__annotations__'):
        print(f"  注解: {types.ThinkingConfig.__annotations__}")

    # 尝试创建一个实例看看接受什么参数
    print("\n尝试创建 ThinkingConfig 实例...")
    try:
        tc1 = types.ThinkingConfig(thinking_budget=1024)
        print("  ✓ thinking_budget=1024 成功")
    except Exception as e:
        print(f"  ✗ thinking_budget=1024 失败: {e}")

    try:
        tc2 = types.ThinkingConfig(thinking_level="low")
        print("  ✓ thinking_level='low' 成功")
    except Exception as e:
        print(f"  ✗ thinking_level='low' 失败: {e}")

# 检查 GenerateContentConfig
print(f"\n是否有 GenerateContentConfig: {hasattr(types, 'GenerateContentConfig')}")
if hasattr(types, 'GenerateContentConfig'):
    print("\nGenerateContentConfig 签名:")
    sig = inspect.signature(types.GenerateContentConfig)
    print(f"  参数: {list(sig.parameters.keys())}")

print("\n" + "=" * 80)
print("完成检查")
print("=" * 80)
