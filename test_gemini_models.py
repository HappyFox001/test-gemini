#!/usr/bin/env python3
"""
Gemini 模型对比测试脚本
简化版：单次测试，对比两个模型的延时和回复质量
"""

import time
import json
from datetime import datetime
from google import genai
from typing import Dict, Tuple

# 初始化客户端
API_KEY = "AIzaSyCxyrthRXXj15jjxwW31IBzCcoVSP36MxY"
client = genai.Client(api_key=API_KEY)

# 测试模型列表
MODELS = [
    "gemini-3-pro-preview",
    "gemini-2.5-pro"
]

# 请求间隔时间（秒）
REQUEST_GAP = 1.0

# 测试提示词（情感交互场景）
TEST_PROMPT = "我最近工作压力特别大，总是担心自己做不好，晚上经常失眠，感觉很焦虑。你能帮我吗？"


def test_model_response(model: str, prompt: str) -> Tuple[str, float]:
    """
    测试单个模型的响应

    Args:
        model: 模型名称
        prompt: 测试提示词

    Returns:
        (响应内容, 响应时间)
    """
    start_time = time.time()

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        end_time = time.time()
        latency = end_time - start_time

        return response.text, latency

    except Exception as e:
        end_time = time.time()
        latency = end_time - start_time
        return f"错误: {str(e)}", latency


def print_separator(char="=", length=80):
    """打印分隔线"""
    print(char * length)


def print_model_result(model: str, response: str, latency: float):
    """打印单个模型的结果"""
    print(f"\n🤖 模型: {model}")
    print(f"⏱️  响应时间: {latency:.3f} 秒")
    print(f"\n💬 回复内容:")
    print("-" * 80)
    print(response)
    print("-" * 80)


def run_comparison_test():
    """运行对比测试"""
    print("\n" + "=" * 80)
    print("🚀 Gemini 模型对比测试")
    print("=" * 80)
    print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 测试模型: {', '.join(MODELS)}")
    print(f"⏱️  请求间隔: {REQUEST_GAP}秒")
    print("=" * 80 + "\n")

    print(f"📝 测试提示词:")
    print(f'"{TEST_PROMPT}"')
    print("\n" + "=" * 80 + "\n")

    # 存储测试结果
    results = {}

    # 测试每个模型
    for idx, model in enumerate(MODELS):
        print(f"⏳ 正在测试 {model}...")
        response, latency = test_model_response(model, TEST_PROMPT)

        print_model_result(model, response, latency)

        results[model] = {
            "response": response,
            "latency": latency
        }

        # 如果不是最后一个模型，等待指定时间
        if idx < len(MODELS) - 1:
            print(f"\n⏸️  等待 {REQUEST_GAP} 秒...\n")
            time.sleep(REQUEST_GAP)

    # 打印总结报告
    print_summary_report(results)

    # 保存结果到文件
    save_results(results)

    return results


def print_summary_report(results: Dict):
    """打印总结报告"""
    print("\n" + "=" * 80)
    print("📊 测试总结报告")
    print("=" * 80 + "\n")

    print("⏱️  响应时间对比:")
    print("-" * 80)
    for model in MODELS:
        if model in results and isinstance(results[model]["latency"], (int, float)):
            latency = results[model]["latency"]
            print(f"{model}: {latency:.3f}秒")
        else:
            print(f"{model}: 无有效数据")

    # 计算速度差异
    latencies = [results[m]["latency"] for m in MODELS if m in results and isinstance(results[m]["latency"], (int, float))]
    if len(latencies) == 2:
        faster = MODELS[0] if latencies[0] < latencies[1] else MODELS[1]
        diff = abs(latencies[0] - latencies[1])
        percent = (diff / max(latencies)) * 100
        print(f"\n⚡ {faster} 更快 {diff:.3f}秒 ({percent:.1f}%)")

    print("\n" + "=" * 80)
    print("\n💡 评估建议:")
    print("-" * 80)
    print("1. 对比两个模型的回复内容质量")
    print("2. 评估情感共鸣能力和共情表达")
    print("3. 检查回复的实用性和可操作性")
    print("4. 注意语气的温暖度和自然度")
    print("5. 综合考虑响应速度和质量的平衡")
    print("=" * 80 + "\n")


def save_results(results: Dict):
    """保存测试结果到JSON文件"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"test_results_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "test_prompt": TEST_PROMPT,
        "request_gap": REQUEST_GAP,
        "results": results
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ 测试结果已保存到: {filename}\n")


if __name__ == "__main__":
    try:
        run_comparison_test()
        print("✨ 测试完成！\n")
    except KeyboardInterrupt:
        print("\n\n⚠️  测试已被用户中断\n")
    except Exception as e:
        print(f"\n\n❌ 测试出错: {str(e)}\n")
