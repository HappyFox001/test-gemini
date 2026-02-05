#!/usr/bin/env python3
"""
Gemini 模型对比测试脚本
测试 Gemini 2.5 Pro 和 3.0 Pro 在延时和情感交互质量上的表现
"""

import time
import json
from datetime import datetime
from google import genai
from typing import Dict, List, Tuple

# 初始化客户端
client = genai.Client()

# 测试模型列表
MODELS = [
    "gemini-2.5-pro",
    "gemini-3.0-pro-preview"
]

# 情感交互测试用例
EMOTION_TEST_CASES = [
    {
        "name": "表达焦虑寻求安慰",
        "prompt": "我最近工作压力特别大，总是担心自己做不好，晚上经常失眠，感觉很焦虑。你能帮我吗？",
        "评估重点": "共情能力、安慰话术、实用建议"
    },
    {
        "name": "失败后需要鼓励",
        "prompt": "我今天的项目演示失败了，在所有同事面前出了丑，感觉自己很没用，不知道该怎么办。",
        "评估重点": "鼓励话术、情绪疏导、积极引导"
    },
    {
        "name": "分享喜悦",
        "prompt": "太开心了！我终于通过了期待已久的面试，这是我梦想的工作！",
        "评估重点": "情感共鸣、真诚祝贺、互动自然度"
    },
    {
        "name": "处理人际冲突",
        "prompt": "我和最好的朋友因为一件小事吵架了，现在很后悔，但不知道该怎么道歉，感觉关系回不到从前了。",
        "评估重点": "情感理解、建议实用性、语气温和度"
    },
    {
        "name": "孤独感倾诉",
        "prompt": "搬到新城市后，我一个朋友都没有，每天下班回家都是一个人，感觉特别孤独。",
        "评估重点": "倾听质量、共情深度、提供支持的方式"
    },
    {
        "name": "自我怀疑",
        "prompt": "我总觉得自己不够好，看到别人都那么优秀，越来越怀疑自己的价值。",
        "评估重点": "认知引导、自我价值肯定、鼓励方式"
    }
]


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


def print_test_header(test_name: str, test_num: int, total: int):
    """打印测试用例标题"""
    print_separator()
    print(f"\n📋 测试 {test_num}/{total}: {test_name}\n")
    print_separator()


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
    print(f"📊 测试用例数: {len(EMOTION_TEST_CASES)}")
    print("=" * 80 + "\n")

    # 存储所有测试结果
    all_results = []

    # 对每个测试用例进行测试
    for idx, test_case in enumerate(EMOTION_TEST_CASES, 1):
        print_test_header(test_case["name"], idx, len(EMOTION_TEST_CASES))

        print(f"\n📝 测试提示词:")
        print(f'"{test_case["prompt"]}"')
        print(f"\n🎯 评估重点: {test_case['评估重点']}\n")

        test_result = {
            "test_name": test_case["name"],
            "prompt": test_case["prompt"],
            "evaluation_focus": test_case["评估重点"],
            "models": {}
        }

        # 测试每个模型
        for model in MODELS:
            print(f"\n⏳ 正在测试 {model}...")
            response, latency = test_model_response(model, test_case["prompt"])

            print_model_result(model, response, latency)

            test_result["models"][model] = {
                "response": response,
                "latency": latency
            }

            # 稍微等待，避免请求过快
            time.sleep(1)

        all_results.append(test_result)
        print("\n")

    # 打印总结报告
    print_summary_report(all_results)

    # 保存结果到文件
    save_results(all_results)

    return all_results


def print_summary_report(results: List[Dict]):
    """打印总结报告"""
    print_separator("=")
    print("\n📊 测试总结报告\n")
    print_separator("=")

    # 计算平均延时
    latency_stats = {model: [] for model in MODELS}

    for result in results:
        for model in MODELS:
            if model in result["models"]:
                latency = result["models"][model]["latency"]
                if isinstance(latency, (int, float)):
                    latency_stats[model].append(latency)

    print("\n⏱️  平均响应时间对比:")
    print("-" * 80)
    for model in MODELS:
        if latency_stats[model]:
            avg_latency = sum(latency_stats[model]) / len(latency_stats[model])
            min_latency = min(latency_stats[model])
            max_latency = max(latency_stats[model])
            print(f"{model}:")
            print(f"  平均: {avg_latency:.3f}秒 | 最小: {min_latency:.3f}秒 | 最大: {max_latency:.3f}秒")
        else:
            print(f"{model}: 无有效数据")

    print("\n" + "=" * 80)
    print("\n💡 评估建议:")
    print("-" * 80)
    print("1. 仔细阅读每个模型的回复内容")
    print("2. 对比情感共鸣能力和共情表达")
    print("3. 评估回复的实用性和可操作性")
    print("4. 注意语气的温暖度和自然度")
    print("5. 综合考虑响应速度和质量的平衡")
    print("=" * 80 + "\n")


def save_results(results: List[Dict]):
    """保存测试结果到JSON文件"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"test_results_{timestamp}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 测试结果已保存到: {filename}\n")


if __name__ == "__main__":
    try:
        run_comparison_test()
        print("✨ 测试完成！\n")
    except KeyboardInterrupt:
        print("\n\n⚠️  测试已被用户中断\n")
    except Exception as e:
        print(f"\n\n❌ 测试出错: {str(e)}\n")
