#!/usr/bin/env python3
"""
Gemini 2.5 Flash 延时测试脚本
测试首 Token 延时 (TTFT)、流式响应时间、多轮对话表现
"""

import os
import time
import json
import requests
from dataclasses import dataclass, field
from typing import Optional, List
import statistics

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============== 配置 ==============

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBipgTphmm7WGfDTkTrmk15egphykiHr_k")
MODEL = "gemini-2.5-flash"  # Gemini 2.5 Flash

# 代理配置
HTTP_PROXY = os.getenv("HTTP_PROXY", "")
HTTPS_PROXY = os.getenv("HTTPS_PROXY", "")
PROXIES = {}
if HTTP_PROXY or HTTPS_PROXY:
    PROXIES = {
        "http": HTTP_PROXY or HTTPS_PROXY,
        "https": HTTPS_PROXY or HTTP_PROXY
    }
    print(f"使用代理: {PROXIES}")

# 测试配置
TEST_ROUNDS = 3  # 每个测试的轮数

# 测试文本
TEST_PROMPTS = {
    "short": "What is 2+2?",
    "medium": "Explain the concept of machine learning in 2-3 sentences.",
    "long": "Write a detailed explanation of how neural networks work, including the concepts of layers, weights, biases, and activation functions. Include an example.",
    "japanese": "日本の四季について簡単に説明してください。",
    "code": "Write a Python function to calculate the Fibonacci sequence up to n terms.",
}

# 多轮对话测试
MULTI_TURN_CONVERSATION = [
    "Hello! I'm learning about space. Can you help me?",
    "What is a black hole?",
    "How are they formed?",
    "Can anything escape from a black hole?",
    "Thank you for the explanation!",
]


@dataclass
class LatencyResult:
    """延时测试结果"""
    test_type: str
    prompt: str
    round_num: int
    success: bool
    ttft: Optional[float] = None  # Time To First Token (秒)
    total_time: Optional[float] = None  # 总响应时间 (秒)
    token_count: Optional[int] = None  # 输出 token 数
    tokens_per_second: Optional[float] = None  # 每秒 token 数
    response_text: Optional[str] = None
    error: Optional[str] = None


@dataclass
class MultiTurnResult:
    """多轮对话测试结果"""
    turn_num: int
    prompt: str
    success: bool
    ttft: Optional[float] = None
    total_time: Optional[float] = None
    response_text: Optional[str] = None
    error: Optional[str] = None


def test_gemini_streaming(prompt: str, conversation_history: List[dict] = None) -> dict:
    """
    测试 Gemini API 流式响应
    返回: {ttft, total_time, token_count, response_text, error}
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:streamGenerateContent"

    headers = {
        "Content-Type": "application/json",
    }

    params = {
        "key": GEMINI_API_KEY,
        "alt": "sse"  # Server-Sent Events for streaming
    }

    # 构建消息
    contents = []
    if conversation_history:
        contents.extend(conversation_history)
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024,
        }
    }

    start_time = time.time()
    first_token_time = None
    response_text = ""
    token_count = 0

    try:
        response = requests.post(
            url,
            headers=headers,
            params=params,
            json=payload,
            stream=True,
            timeout=60,
            proxies=PROXIES
        )

        if response.status_code != 200:
            return {
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "ttft": None,
                "total_time": None,
                "token_count": None,
                "response_text": None
            }

        # 处理 SSE 流
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    json_str = line_str[6:]  # 移除 "data: " 前缀
                    try:
                        data = json.loads(json_str)

                        # 提取文本
                        candidates = data.get("candidates", [])
                        if candidates:
                            content = candidates[0].get("content", {})
                            parts = content.get("parts", [])
                            for part in parts:
                                text = part.get("text", "")
                                if text:
                                    if first_token_time is None:
                                        first_token_time = time.time()
                                    response_text += text
                                    # 粗略估计 token 数（按空格和标点分割）
                                    token_count += len(text.split())
                    except json.JSONDecodeError:
                        continue

        end_time = time.time()

        return {
            "ttft": first_token_time - start_time if first_token_time else None,
            "total_time": end_time - start_time,
            "token_count": token_count,
            "response_text": response_text,
            "error": None
        }

    except requests.exceptions.Timeout:
        return {
            "error": "Request timeout",
            "ttft": None,
            "total_time": None,
            "token_count": None,
            "response_text": None
        }
    except Exception as e:
        return {
            "error": str(e),
            "ttft": None,
            "total_time": None,
            "token_count": None,
            "response_text": None
        }


def test_gemini_non_streaming(prompt: str, conversation_history: List[dict] = None) -> dict:
    """
    测试 Gemini API 非流式响应
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

    headers = {
        "Content-Type": "application/json",
    }

    params = {
        "key": GEMINI_API_KEY,
    }

    # 构建消息
    contents = []
    if conversation_history:
        contents.extend(conversation_history)
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024,
        }
    }

    start_time = time.time()

    try:
        response = requests.post(
            url,
            headers=headers,
            params=params,
            json=payload,
            timeout=60,
            proxies=PROXIES
        )

        end_time = time.time()

        if response.status_code != 200:
            return {
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "ttft": None,
                "total_time": None,
                "token_count": None,
                "response_text": None
            }

        data = response.json()

        # 提取文本
        response_text = ""
        candidates = data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            for part in parts:
                response_text += part.get("text", "")

        # 获取 token 计数
        usage = data.get("usageMetadata", {})
        token_count = usage.get("candidatesTokenCount", len(response_text.split()))

        return {
            "ttft": end_time - start_time,  # 非流式时 TTFT = 总时间
            "total_time": end_time - start_time,
            "token_count": token_count,
            "response_text": response_text,
            "error": None
        }

    except Exception as e:
        return {
            "error": str(e),
            "ttft": None,
            "total_time": None,
            "token_count": None,
            "response_text": None
        }


def run_single_prompt_tests() -> List[LatencyResult]:
    """运行单轮提示测试"""
    results = []

    print("=" * 70)
    print("单轮提示延时测试 (流式)")
    print("=" * 70)
    print()

    for test_type, prompt in TEST_PROMPTS.items():
        print(f"测试 [{test_type}]: {prompt[:50]}...")

        for round_num in range(1, TEST_ROUNDS + 1):
            result_data = test_gemini_streaming(prompt)

            result = LatencyResult(
                test_type=test_type,
                prompt=prompt,
                round_num=round_num,
                success=result_data["error"] is None,
                ttft=result_data["ttft"],
                total_time=result_data["total_time"],
                token_count=result_data["token_count"],
                response_text=result_data["response_text"],
                error=result_data["error"]
            )

            if result.ttft and result.total_time and result.token_count:
                result.tokens_per_second = result.token_count / result.total_time

            results.append(result)

            if result.success:
                print(f"  Round {round_num}: TTFT={result.ttft:.3f}s, Total={result.total_time:.3f}s, Tokens={result.token_count}")
            else:
                print(f"  Round {round_num}: 失败 - {result.error}")

            # 轮次间隔
            if round_num < TEST_ROUNDS:
                time.sleep(0.5)

        print()

    return results


def run_multi_turn_test() -> List[MultiTurnResult]:
    """运行多轮对话测试"""
    results = []
    conversation_history = []

    print("=" * 70)
    print("多轮对话延时测试")
    print("=" * 70)
    print()

    for turn_num, prompt in enumerate(MULTI_TURN_CONVERSATION, 1):
        print(f"Turn {turn_num}: {prompt[:50]}...")

        result_data = test_gemini_streaming(prompt, conversation_history)

        result = MultiTurnResult(
            turn_num=turn_num,
            prompt=prompt,
            success=result_data["error"] is None,
            ttft=result_data["ttft"],
            total_time=result_data["total_time"],
            response_text=result_data["response_text"],
            error=result_data["error"]
        )

        results.append(result)

        if result.success:
            print(f"  TTFT={result.ttft:.3f}s, Total={result.total_time:.3f}s")
            print(f"  Response: {result.response_text[:100]}..." if len(result.response_text) > 100 else f"  Response: {result.response_text}")

            # 更新对话历史
            conversation_history.append({
                "role": "user",
                "parts": [{"text": prompt}]
            })
            conversation_history.append({
                "role": "model",
                "parts": [{"text": result.response_text}]
            })
        else:
            print(f"  失败 - {result.error}")

        print()
        time.sleep(0.5)

    return results


def run_streaming_vs_non_streaming_test() -> dict:
    """比较流式和非流式响应"""
    print("=" * 70)
    print("流式 vs 非流式对比测试")
    print("=" * 70)
    print()

    test_prompt = "Explain quantum computing in simple terms."

    results = {
        "streaming": [],
        "non_streaming": []
    }

    print("测试流式响应...")
    for i in range(TEST_ROUNDS):
        result = test_gemini_streaming(test_prompt)
        if result["error"] is None:
            results["streaming"].append({
                "ttft": result["ttft"],
                "total_time": result["total_time"]
            })
            print(f"  Round {i+1}: TTFT={result['ttft']:.3f}s, Total={result['total_time']:.3f}s")
        else:
            print(f"  Round {i+1}: 失败 - {result['error']}")
        time.sleep(0.5)

    print()
    print("测试非流式响应...")
    for i in range(TEST_ROUNDS):
        result = test_gemini_non_streaming(test_prompt)
        if result["error"] is None:
            results["non_streaming"].append({
                "ttft": result["ttft"],
                "total_time": result["total_time"]
            })
            print(f"  Round {i+1}: Total={result['total_time']:.3f}s")
        else:
            print(f"  Round {i+1}: 失败 - {result['error']}")
        time.sleep(0.5)

    print()

    return results


def generate_report(single_results: List[LatencyResult],
                   multi_results: List[MultiTurnResult],
                   streaming_comparison: dict):
    """生成测试报告"""
    report = []
    report.append("# Gemini 2.5 Flash 延时测试报告")
    report.append("")
    report.append(f"**模型**: {MODEL}")
    report.append(f"**测试轮数**: {TEST_ROUNDS}")
    report.append("")

    # 单轮测试汇总
    report.append("## 1. 单轮提示延时测试 (流式)")
    report.append("")
    report.append("| 测试类型 | 平均TTFT | 最小TTFT | 最大TTFT | 平均总时间 | 平均TPS | 成功率 |")
    report.append("|----------|----------|----------|----------|------------|---------|--------|")

    for test_type in TEST_PROMPTS.keys():
        type_results = [r for r in single_results if r.test_type == test_type]
        successful = [r for r in type_results if r.success]

        if successful:
            ttfts = [r.ttft for r in successful if r.ttft]
            totals = [r.total_time for r in successful if r.total_time]
            tps_list = [r.tokens_per_second for r in successful if r.tokens_per_second]

            avg_ttft = statistics.mean(ttfts) if ttfts else 0
            min_ttft = min(ttfts) if ttfts else 0
            max_ttft = max(ttfts) if ttfts else 0
            avg_total = statistics.mean(totals) if totals else 0
            avg_tps = statistics.mean(tps_list) if tps_list else 0
            success_rate = len(successful) / len(type_results) * 100

            report.append(f"| {test_type} | {avg_ttft:.3f}s | {min_ttft:.3f}s | {max_ttft:.3f}s | {avg_total:.3f}s | {avg_tps:.1f} | {success_rate:.0f}% |")
        else:
            report.append(f"| {test_type} | N/A | N/A | N/A | N/A | N/A | 0% |")

    report.append("")

    # 多轮对话测试
    report.append("## 2. 多轮对话延时测试")
    report.append("")
    report.append("| Turn | Prompt | TTFT | 总时间 | 状态 |")
    report.append("|------|--------|------|--------|------|")

    for r in multi_results:
        status = "✓ 成功" if r.success else f"✗ {r.error}"
        ttft_str = f"{r.ttft:.3f}s" if r.ttft else "N/A"
        total_str = f"{r.total_time:.3f}s" if r.total_time else "N/A"
        prompt_short = r.prompt[:30] + "..." if len(r.prompt) > 30 else r.prompt
        report.append(f"| {r.turn_num} | {prompt_short} | {ttft_str} | {total_str} | {status} |")

    # 多轮统计
    successful_multi = [r for r in multi_results if r.success]
    if successful_multi:
        avg_ttft = statistics.mean([r.ttft for r in successful_multi if r.ttft])
        avg_total = statistics.mean([r.total_time for r in successful_multi if r.total_time])
        report.append("")
        report.append(f"**多轮对话平均 TTFT**: {avg_ttft:.3f}s")
        report.append(f"**多轮对话平均总时间**: {avg_total:.3f}s")

    report.append("")

    # 流式 vs 非流式对比
    report.append("## 3. 流式 vs 非流式对比")
    report.append("")
    report.append("| 模式 | 平均TTFT | 平均总时间 |")
    report.append("|------|----------|------------|")

    if streaming_comparison["streaming"]:
        avg_ttft = statistics.mean([r["ttft"] for r in streaming_comparison["streaming"]])
        avg_total = statistics.mean([r["total_time"] for r in streaming_comparison["streaming"]])
        report.append(f"| 流式 | {avg_ttft:.3f}s | {avg_total:.3f}s |")

    if streaming_comparison["non_streaming"]:
        avg_total = statistics.mean([r["total_time"] for r in streaming_comparison["non_streaming"]])
        report.append(f"| 非流式 | {avg_total:.3f}s | {avg_total:.3f}s |")

    report.append("")

    # 关键指标说明
    report.append("## 指标说明")
    report.append("")
    report.append("- **TTFT (Time To First Token)**: 从发送请求到收到第一个 token 的时间")
    report.append("- **总时间**: 从发送请求到收到完整响应的时间")
    report.append("- **TPS (Tokens Per Second)**: 每秒生成的 token 数")
    report.append("")

    # 结论
    report.append("## 结论")
    report.append("")

    # 计算整体统计
    all_successful = [r for r in single_results if r.success]
    if all_successful:
        overall_avg_ttft = statistics.mean([r.ttft for r in all_successful if r.ttft])
        overall_avg_total = statistics.mean([r.total_time for r in all_successful if r.total_time])
        report.append(f"- **整体平均 TTFT**: {overall_avg_ttft:.3f}s")
        report.append(f"- **整体平均总时间**: {overall_avg_total:.3f}s")
        report.append(f"- **测试成功率**: {len(all_successful)}/{len(single_results)} ({len(all_successful)/len(single_results)*100:.0f}%)")

    report.append("")

    return "\n".join(report)


def main():
    """主函数"""
    print("=" * 70)
    print(f"Gemini 2.5 Flash 延时测试")
    print(f"模型: {MODEL}")
    print("=" * 70)
    print()

    # 1. 单轮提示测试
    single_results = run_single_prompt_tests()

    # 2. 多轮对话测试
    multi_results = run_multi_turn_test()

    # 3. 流式 vs 非流式对比
    streaming_comparison = run_streaming_vs_non_streaming_test()

    # 生成报告
    report = generate_report(single_results, multi_results, streaming_comparison)

    # 保存报告
    report_file = "/home/andy/work/voice-test/output/gemini_latency_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print("=" * 70)
    print("测试报告")
    print("=" * 70)
    print(report)
    print()
    print(f"报告已保存到: {report_file}")


if __name__ == "__main__":
    main()
