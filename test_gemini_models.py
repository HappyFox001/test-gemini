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
from google.genai import types
from typing import Dict, Tuple, Optional

# 初始化客户端
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("请设置环境变量 GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

# 测试模型配置列表（串行测试，一个完成后才测下一个）
# 每个配置包含：name (显示名称), model (模型ID), thinking_level (thinking配置)
# 注意：Gemini 3 Pro 无法完全关闭 thinking，最低只能设置为 LOW
MODEL_CONFIGS = [
    # Flash 模型：测试 LOW 和 HIGH thinking（Flash 可以设置为 0 来关闭）
    {
        "name": "gemini-3-flash (thinking OFF)",
        "model": "gemini-3-flash-preview",
        "thinking_budget": 0  # 设置为 0 关闭 thinking
    },
    {
        "name": "gemini-3-flash (thinking LOW)",
        "model": "gemini-3-flash-preview",
        "thinking_level": "low"  # 使用小写字符串
    },
    {
        "name": "gemini-3-flash (thinking HIGH)",
        "model": "gemini-3-flash-preview",
        "thinking_level": "high"  # 使用小写字符串
    },

    # Pro 模型：只能测试 LOW 和 HIGH（无法完全关闭）
    {
        "name": "gemini-3-pro (thinking LOW)",
        "model": "gemini-3-pro-preview",
        "thinking_level": "low"  # 使用小写字符串
    },
    {
        "name": "gemini-3-pro (thinking HIGH - default)",
        "model": "gemini-3-pro-preview",
        "thinking_level": "high"  # 使用小写字符串，这是默认值
    },
]

# 两轮对话的提示词
PROMPTS = [
    "你好，我最近工作压力很大，感觉很焦虑。",
    "谢谢你的建议，我该如何开始改善这个状况呢？"
]

# 不使用硬编码的等待时间，完全依赖同步执行
# 每个请求会等待完全完成后才开始下一个


def test_model_with_timing(model: str, prompt: str, thinking_level = None, thinking_budget: Optional[int] = None) -> Dict:
    """
    测试单个模型的响应，记录详细时间数据

    Args:
        model: 模型ID
        prompt: 提示词
        thinking_level: thinking 级别（types.ThinkingLevel.LOW 或 types.ThinkingLevel.HIGH）
        thinking_budget: thinking 预算（整数，0 表示关闭）

    Returns:
        {
            'first_token_time': 首字延时（秒）,
            'total_time': 总响应时间（秒）,
            'response_length': 响应字符数,
            'response_text': 响应内容
        }
    """
    start_time = time.time()
    first_chunk_time = None  # 第一个 chunk 到达时间
    first_token_time = None  # 第一个文本 token 到达时间
    response_text = ""
    chunk_count = 0

    try:
        # 构建请求参数
        request_params = {
            "model": model,
            "contents": prompt,
        }

        # 如果提供了 thinking 配置，添加到 config 中
        if thinking_level is not None or thinking_budget is not None:
            thinking_config_kwargs = {}

            if thinking_budget is not None:
                thinking_config_kwargs["thinking_budget"] = thinking_budget
                print(f"    [调试] Thinking 配置: thinking_budget={thinking_budget}")
            elif thinking_level is not None:
                thinking_config_kwargs["thinking_level"] = thinking_level
                print(f"    [调试] Thinking 配置: thinking_level={thinking_level}")

            request_params["config"] = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(**thinking_config_kwargs)
            )

        # 使用流式响应来获取首字延时
        response = client.models.generate_content_stream(**request_params)

        # 接收流式响应
        for chunk in response:
            chunk_count += 1
            current_time = time.time() - start_time

            # 记录第一个 chunk 到达时间（无论是否有文本）
            if first_chunk_time is None:
                first_chunk_time = current_time
                print(f"    [调试] 首个 chunk 到达: {first_chunk_time:.3f}秒")

            # 提取文本内容
            if hasattr(chunk, 'candidates') and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        # 只提取文本部分，忽略 thought_signature 等
                        if hasattr(part, 'text') and part.text:
                            # 记录首字延时（第一个包含文本的 chunk 到达时间）
                            if first_token_time is None:
                                first_token_time = current_time
                                print(f"    [调试] 首个文本到达: {first_token_time:.3f}秒 (chunk #{chunk_count})")
                            response_text += part.text

        total_time = time.time() - start_time
        print(f"    [调试] 总 chunks: {chunk_count}, 总时间: {total_time:.3f}秒")

        return {
            'first_chunk_time': first_chunk_time or 0,
            'first_token_time': first_token_time or 0,
            'total_time': total_time,
            'response_length': len(response_text),
            'chunk_count': chunk_count,
            'response_text': response_text
        }

    except Exception as e:
        total_time = time.time() - start_time
        error_msg = str(e)
        print(f"    [错误] {error_msg}")
        return {
            'first_chunk_time': 0,
            'first_token_time': 0,
            'total_time': total_time,
            'response_length': 0,
            'chunk_count': 0,
            'response_text': f"错误: {error_msg}",
            'error': error_msg
        }


def test_network_latency():
    """测试网络延迟"""
    print("\n🌐 测试网络延迟到 Google API...")
    try:
        import urllib.request
        start = time.time()
        urllib.request.urlopen('https://generativelanguage.googleapis.com', timeout=10)
        latency = time.time() - start
        print(f"   网络延迟: {latency:.3f}秒")
        if latency > 1:
            print(f"   ⚠️  网络延迟较高 (>{latency:.1f}秒)")
        return latency
    except Exception as e:
        print(f"   ❌ 网络测试失败: {e}")
        return None


def run_performance_test():
    """运行性能测试"""
    print("\n" + "=" * 80)
    print("🚀 Gemini 模型延时性能测试 - Thinking 配置对比")
    print("=" * 80)
    print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 测试配置: {len(MODEL_CONFIGS)} 个")
    for config in MODEL_CONFIGS:
        print(f"    - {config['name']}")
    print(f"💬 对话轮数: {len(PROMPTS)}")
    print(f"🔄 执行模式: 完全串行 (每个请求完成后才开始下一个)")
    print("=" * 80)

    # 测试网络延迟
    test_network_latency()
    print("=" * 80 + "\n")

    # 存储所有结果
    all_results = {}

    # 串行测试每个模型配置（一个完成后才开始下一个）
    for config_idx, config in enumerate(MODEL_CONFIGS, 1):
        config_name = config['name']
        model_id = config['model']
        thinking_level = config.get('thinking_level')
        thinking_budget = config.get('thinking_budget')

        print(f"📊 测试配置 {config_idx}/{len(MODEL_CONFIGS)}: {config_name}")
        print(f"   模型: {model_id}")
        if thinking_level:
            print(f"   Thinking Level: {thinking_level}")
        if thinking_budget is not None:
            print(f"   Thinking Budget: {thinking_budget}")
        print("-" * 80)

        model_results = {
            'model': model_id,
            'thinking_level': thinking_level,
            'thinking_budget': thinking_budget,
            'conversations': [],
            'total_length': 0,
            'total_time': 0
        }

        # 进行两轮对话
        for round_num, prompt in enumerate(PROMPTS, 1):
            print(f"\n第 {round_num} 轮对话...")
            print(f"提示词: {prompt}")

            # 测试响应
            result = test_model_with_timing(
                model_id,
                prompt,
                thinking_level=thinking_level,
                thinking_budget=thinking_budget
            )

            # 打印结果
            print(f"├─ 首 chunk 延时: {result.get('first_chunk_time', 0):.3f}秒")
            print(f"├─ 首文本延时: {result['first_token_time']:.3f}秒")
            print(f"├─ 总响应时间: {result['total_time']:.3f}秒")
            print(f"├─ 响应长度: {result['response_length']}字符")
            print(f"└─ Chunks 数量: {result.get('chunk_count', 0)}")

            model_results['conversations'].append(result)
            model_results['total_length'] += result['response_length']
            model_results['total_time'] += result['total_time']

            # 不添加人工等待，直接进入下一轮（自然串行执行）
            if round_num < len(PROMPTS):
                print()

        all_results[config_name] = model_results

        # 不添加人工等待，当前配置完成后自动开始下一个
        if config_idx < len(MODEL_CONFIGS):
            print(f"\n✅ {config_name} 测试完成，开始下一个配置...\n")

        print("=" * 80 + "\n")

    # 打印对比表格
    print_comparison_table(all_results)

    # 保存结果
    save_results(all_results)

    return all_results


def print_comparison_table(results: Dict):
    """打印性能对比表格"""
    print("\n" + "=" * 120)
    print("📊 性能对比总结 - Thinking 配置影响")
    print("=" * 120 + "\n")

    # 表头
    header = f"{'配置名称':<35} {'首字(R1)':<12} {'总时(R1)':<12} {'长度(R1)':<10} {'首字(R2)':<12} {'总时(R2)':<12} {'长度(R2)':<10}"
    print(header)
    print("-" * 120)

    # 遍历所有配置并打印结果
    for config_name, config_results in results.items():
        convs = config_results.get('conversations', [])

        # 第一轮数据
        conv1 = convs[0] if len(convs) > 0 else {}
        first_token_r1 = conv1.get('first_token_time', 0)
        total_time_r1 = conv1.get('total_time', 0)
        length_r1 = conv1.get('response_length', 0)

        # 第二轮数据
        conv2 = convs[1] if len(convs) > 1 else {}
        first_token_r2 = conv2.get('first_token_time', 0)
        total_time_r2 = conv2.get('total_time', 0)
        length_r2 = conv2.get('response_length', 0)

        # 打印一行数据
        row = f"{config_name:<35} {first_token_r1:>7.3f}秒   {total_time_r1:>7.3f}秒   {length_r1:>7}字   {first_token_r2:>7.3f}秒   {total_time_r2:>7.3f}秒   {length_r2:>7}字"
        print(row)

    print("-" * 120)

    # 打印分组对比
    print("\n📈 分组对比分析:")
    print("-" * 120)

    # Flash 模型对比
    print("\n🔵 Flash 模型 - Thinking 配置对比:")
    flash_configs = ["gemini-3-flash (thinking OFF)", "gemini-3-flash (thinking LOW)", "gemini-3-flash (thinking HIGH)"]
    flash_results_list = []
    for config_name in flash_configs:
        config_data = results.get(config_name, {})
        if config_data:
            print(f"   {config_name:<35} - 总时间: {config_data.get('total_time', 0):>6.3f}秒, 总长度: {config_data.get('total_length', 0):>5}字")
            flash_results_list.append((config_name, config_data))

    # 计算 Flash 的时间差异
    if len(flash_results_list) >= 2:
        flash_off = results.get("gemini-3-flash (thinking OFF)", {})
        flash_high = results.get("gemini-3-flash (thinking HIGH)", {})
        if flash_off and flash_high and flash_off.get('total_time', 0) > 0:
            time_diff = ((flash_high.get('total_time', 0) - flash_off.get('total_time', 0)) / flash_off.get('total_time', 1)) * 100
            print(f"   时间差异: {time_diff:+.1f}% (HIGH vs OFF)")

    # Pro 模型对比
    print("\n🟣 Pro 模型 - Thinking 配置对比 (注意：Pro 无法完全关闭 thinking):")
    pro_configs = ["gemini-3-pro (thinking LOW)", "gemini-3-pro (thinking HIGH - default)"]
    for config_name in pro_configs:
        config_data = results.get(config_name, {})
        if config_data:
            print(f"   {config_name:<35} - 总时间: {config_data.get('total_time', 0):>6.3f}秒, 总长度: {config_data.get('total_length', 0):>5}字")

    print("\n" + "=" * 120 + "\n")


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
                    'first_chunk_time': conv.get('first_chunk_time', 0),
                    'first_token_time': conv.get('first_token_time', 0),
                    'total_time': conv.get('total_time', 0),
                    'response_length': conv.get('response_length', 0),
                    'chunk_count': conv.get('chunk_count', 0)
                }
                for conv in data['conversations']
            ]
        }

    output = {
        'timestamp': timestamp,
        'test_type': 'thinking_config_comparison',
        'configurations': [
            {
                'name': config['name'],
                'model': config['model'],
                'thinking_level': config.get('thinking_level'),
                'thinking_budget': config.get('thinking_budget')
            }
            for config in MODEL_CONFIGS
        ],
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
