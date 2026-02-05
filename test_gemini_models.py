#!/usr/bin/env python3
"""
Gemini 模型延时性能测试
测试首字延时、总响应时间、对话间隔
"""

import time
import json
import os
from datetime import datetime
from google import genai
from typing import Dict, Tuple

# 初始化客户端
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("请设置环境变量 GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

# 测试模型列表
MODELS = [
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash"
]

# 两轮对话的提示词
PROMPTS = [
    "你好，我最近工作压力很大，感觉很焦虑。",
    "谢谢你的建议，我该如何开始改善这个状况呢？"
]

# 对话间隔时间（秒）
CONVERSATION_GAP = 0.5


def test_model_with_timing(model: str, prompt: str) -> Dict:
    """
    测试单个模型的响应，记录详细时间数据

    Returns:
        {
            'first_token_time': 首字延时（秒）,
            'total_time': 总响应时间（秒）,
            'response_length': 响应字符数,
            'response_text': 响应内容
        }
    """
    start_time = time.time()
    first_token_time = None
    response_text = ""

    try:
        # 使用流式响应来获取首字延时
        response = client.models.generate_content_stream(
            model=model,
            contents=prompt,
        )

        # 接收流式响应
        for chunk in response:
            # 记录首字延时（第一个 chunk 到达时间）
            if first_token_time is None:
                first_token_time = time.time() - start_time

            # 提取文本内容
            if hasattr(chunk, 'candidates') and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        # 只提取文本部分，忽略 thought_signature 等
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text

        total_time = time.time() - start_time

        return {
            'first_token_time': first_token_time or 0,
            'total_time': total_time,
            'response_length': len(response_text),
            'response_text': response_text
        }

    except Exception as e:
        total_time = time.time() - start_time
        return {
            'first_token_time': 0,
            'total_time': total_time,
            'response_length': 0,
            'response_text': f"错误: {str(e)}",
            'error': str(e)
        }


def run_performance_test():
    """运行性能测试"""
    print("\n" + "=" * 80)
    print("🚀 Gemini 模型延时性能测试")
    print("=" * 80)
    print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 测试模型: {', '.join(MODELS)}")
    print(f"💬 对话轮数: {len(PROMPTS)}")
    print(f"⏱️  对话间隔: {CONVERSATION_GAP}秒")
    print("=" * 80 + "\n")

    # 存储所有结果
    all_results = {}

    # 测试每个模型
    for model in MODELS:
        print(f"📊 测试模型: {model}")
        print("-" * 80)

        model_results = {
            'conversations': [],
            'total_length': 0,
            'total_time': 0,
            'gaps': []
        }

        # 进行两轮对话
        for round_num, prompt in enumerate(PROMPTS, 1):
            print(f"\n第 {round_num} 轮对话...")
            print(f"提示词: {prompt}")

            # 测试响应
            result = test_model_with_timing(model, prompt)

            # 打印结果
            print(f"├─ 首字延时: {result['first_token_time']:.3f}秒")
            print(f"├─ 总响应时间: {result['total_time']:.3f}秒")
            print(f"└─ 响应长度: {result['response_length']}字符")

            model_results['conversations'].append(result)
            model_results['total_length'] += result['response_length']
            model_results['total_time'] += result['total_time']

            # 如果不是最后一轮，等待并记录间隔
            if round_num < len(PROMPTS):
                gap_start = time.time()
                time.sleep(CONVERSATION_GAP)
                actual_gap = time.time() - gap_start
                model_results['gaps'].append(actual_gap)
                print(f"\n⏸️  对话间隔: {actual_gap:.3f}秒")

        all_results[model] = model_results
        print("\n" + "=" * 80 + "\n")

    # 打印对比表格
    print_comparison_table(all_results)

    # 保存结果
    save_results(all_results)

    return all_results


def print_comparison_table(results: Dict):
    """打印性能对比表格"""
    print("\n" + "=" * 80)
    print("📊 性能对比总结")
    print("=" * 80 + "\n")

    # 表头
    print(f"{'指标':<20} {'gemini-3-flash':<20} {'gemini-3-pro':<20}")
    print("-" * 80)

    # 提取两个模型的结果
    flash_key = "gemini-3-flash-preview"
    pro_key = "gemini-3-pro-preview"

    flash_results = results.get(flash_key, {})
    pro_results = results.get(pro_key, {})

    # 第一轮对话数据
    flash_conv1 = flash_results.get('conversations', [{}])[0]
    pro_conv1 = pro_results.get('conversations', [{}])[0]

    print(f"{'[第1轮] 首字延时':<20} {flash_conv1.get('first_token_time', 0):.3f}秒{'':<13} {pro_conv1.get('first_token_time', 0):.3f}秒")
    print(f"{'[第1轮] 总时间':<20} {flash_conv1.get('total_time', 0):.3f}秒{'':<13} {pro_conv1.get('total_time', 0):.3f}秒")
    print(f"{'[第1轮] 响应长度':<20} {flash_conv1.get('response_length', 0)}字符{'':<13} {pro_conv1.get('response_length', 0)}字符")

    # 对话间隔
    flash_gap = flash_results.get('gaps', [0])[0] if flash_results.get('gaps') else 0
    pro_gap = pro_results.get('gaps', [0])[0] if pro_results.get('gaps') else 0
    print(f"{'对话间隔':<20} {flash_gap:.3f}秒{'':<13} {pro_gap:.3f}秒")

    # 第二轮对话数据
    flash_conv2 = flash_results.get('conversations', [{}])[1] if len(flash_results.get('conversations', [])) > 1 else {}
    pro_conv2 = pro_results.get('conversations', [{}])[1] if len(pro_results.get('conversations', [])) > 1 else {}

    print(f"{'[第2轮] 首字延时':<20} {flash_conv2.get('first_token_time', 0):.3f}秒{'':<13} {pro_conv2.get('first_token_time', 0):.3f}秒")
    print(f"{'[第2轮] 总时间':<20} {flash_conv2.get('total_time', 0):.3f}秒{'':<13} {pro_conv2.get('total_time', 0):.3f}秒")
    print(f"{'[第2轮] 响应长度':<20} {flash_conv2.get('response_length', 0)}字符{'':<13} {pro_conv2.get('response_length', 0)}字符")

    print("-" * 80)

    # 总计
    print(f"{'总响应长度':<20} {flash_results.get('total_length', 0)}字符{'':<13} {pro_results.get('total_length', 0)}字符")
    print(f"{'总响应时间':<20} {flash_results.get('total_time', 0):.3f}秒{'':<13} {pro_results.get('total_time', 0):.3f}秒")

    # 计算平均速度（字符/秒）
    flash_speed = flash_results.get('total_length', 0) / flash_results.get('total_time', 1) if flash_results.get('total_time', 0) > 0 else 0
    pro_speed = pro_results.get('total_length', 0) / pro_results.get('total_time', 1) if pro_results.get('total_time', 0) > 0 else 0

    print(f"{'平均速度':<20} {flash_speed:.1f}字符/秒{'':<8} {pro_speed:.1f}字符/秒")

    print("\n" + "=" * 80 + "\n")


def save_results(results: Dict):
    """保存测试结果到JSON文件"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"performance_test_{timestamp}.json"

    # 简化输出，只保留时间数据
    simplified_results = {}

    for model, data in results.items():
        simplified_results[model] = {
            'total_length': data['total_length'],
            'total_time': data['total_time'],
            'conversations': [
                {
                    'first_token_time': conv.get('first_token_time', 0),
                    'total_time': conv.get('total_time', 0),
                    'response_length': conv.get('response_length', 0)
                }
                for conv in data['conversations']
            ],
            'gaps': data.get('gaps', [])
        }

    output = {
        'timestamp': timestamp,
        'test_type': 'latency_performance',
        'models': MODELS,
        'conversation_rounds': len(PROMPTS),
        'results': simplified_results
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ 测试结果已保存到: {filename}\n")


if __name__ == "__main__":
    try:
        run_performance_test()
        print("✨ 测试完成！\n")
    except KeyboardInterrupt:
        print("\n\n⚠️  测试已被用户中断\n")
    except Exception as e:
        print(f"\n\n❌ 测试出错: {str(e)}\n")
        import traceback
        traceback.print_exc()
