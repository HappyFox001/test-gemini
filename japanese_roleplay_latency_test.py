#!/usr/bin/env python3
"""
日语 Roleplay 对话延时测试
测试关闭 thinking 后的最低延时表现
"""

import os
import time
import json
import requests
from dataclasses import dataclass
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

# 测试模型 (gemini-3-pro-preview 暂时移除，API 响应异常)
MODELS = [
    "gemini-3-flash-preview",
    # "gemini-3-pro-preview",  # API 返回空响应，暂时跳过
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

# 支持 thinking 的模型 (可以关闭)
THINKING_MODELS_OPTIONAL = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
]

# 只能使用 thinking 的模型 (不能关闭)
THINKING_MODELS_REQUIRED = [
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
]

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

# 测试轮数
TEST_ROUNDS = 3

# 日语 Roleplay 对话场景
ROLEPLAY_SYSTEM = "あなたは「さくら」という名前の20歳の女子大生です。明るくて元気な性格で、話し方はカジュアルで親しみやすいです。短く自然な返答をしてください。1-2文で返答してください。"

# 测试对话 (简单日常对话)
TEST_DIALOGUES = [
    "おはよう！今日の調子はどう？",
    "今日は何する予定？",
    "週末一緒に遊ばない？",
]

# 多轮对话测试
MULTI_TURN_DIALOGUES = [
    "やっほー！さくらちゃん元気？",
    "今日暇？",
    "じゃあカフェ行こうよ",
    "何飲む？",
    "ありがとう！また遊ぼうね",
]


@dataclass
class LatencyResult:
    """延时结果"""
    model: str
    thinking_enabled: bool
    prompt: str
    round_num: int
    success: bool
    ttft: Optional[float] = None
    total_time: Optional[float] = None
    response_text: Optional[str] = None
    error: Optional[str] = None


def test_gemini_streaming(
    model: str,
    prompt: str,
    system_instruction: str = None,
    conversation_history: List[dict] = None,
    disable_thinking: bool = True,
    max_tokens: int = 128
) -> dict:
    """
    测试 Gemini API 流式响应
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"

    headers = {
        "Content-Type": "application/json",
    }

    params = {
        "key": GEMINI_API_KEY,
        "alt": "sse"
    }

    # 构建消息
    contents = []
    if conversation_history:
        contents.extend(conversation_history)
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })

    # 构建 payload
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.8,
            "maxOutputTokens": max_tokens,
        }
    }

    # 添加 system instruction
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }

    # 关闭 thinking (仅对支持的模型)
    if disable_thinking and any(m in model for m in ["2.5", "3-flash", "3-pro"]):
        payload["generationConfig"]["thinkingConfig"] = {
            "thinkingBudget": 0
        }

    start_time = time.time()
    first_token_time = None
    response_text = ""

    try:
        response = requests.post(
            url,
            headers=headers,
            params=params,
            json=payload,
            stream=True,
            timeout=30,
            proxies=PROXIES
        )

        if response.status_code != 200:
            return {
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "ttft": None,
                "total_time": None,
                "response_text": None
            }

        # 处理 SSE 流
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    json_str = line_str[6:]
                    try:
                        data = json.loads(json_str)
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
                    except json.JSONDecodeError:
                        continue

        end_time = time.time()

        return {
            "ttft": first_token_time - start_time if first_token_time else None,
            "total_time": end_time - start_time,
            "response_text": response_text,
            "error": None
        }

    except requests.exceptions.Timeout:
        return {
            "error": "Request timeout",
            "ttft": None,
            "total_time": None,
            "response_text": None
        }
    except Exception as e:
        return {
            "error": str(e),
            "ttft": None,
            "total_time": None,
            "response_text": None
        }


def run_single_turn_tests() -> List[LatencyResult]:
    """单轮对话测试"""
    results = []

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"模型: {model}")
        print(f"{'='*60}")

        # 测试模式:
        # - 可选 thinking 的模型: 测试 ON 和 OFF
        # - 必须 thinking 的模型: 只测试 ON
        # - 不支持 thinking 的模型: 只测试 OFF
        if model in THINKING_MODELS_OPTIONAL:
            thinking_modes = [True, False]
        elif model in THINKING_MODELS_REQUIRED:
            thinking_modes = [True]  # 只能开启
        else:
            thinking_modes = [False]  # 不支持 thinking

        for thinking_enabled in thinking_modes:
            mode_str = "thinking ON" if thinking_enabled else "thinking OFF"
            print(f"\n--- {mode_str} ---")

            for prompt in TEST_DIALOGUES:
                print(f"\n提示: {prompt}")

                for round_num in range(1, TEST_ROUNDS + 1):
                    result_data = test_gemini_streaming(
                        model=model,
                        prompt=prompt,
                        system_instruction=ROLEPLAY_SYSTEM,
                        disable_thinking=not thinking_enabled,
                        max_tokens=128
                    )

                    result = LatencyResult(
                        model=model,
                        thinking_enabled=thinking_enabled,
                        prompt=prompt,
                        round_num=round_num,
                        success=result_data["error"] is None,
                        ttft=result_data["ttft"],
                        total_time=result_data["total_time"],
                        response_text=result_data["response_text"],
                        error=result_data["error"]
                    )
                    results.append(result)

                    if result.success and result.ttft is not None:
                        resp_preview = result.response_text[:30] if result.response_text else ""
                        print(f"  R{round_num}: TTFT={result.ttft:.3f}s, Total={result.total_time:.3f}s | {resp_preview}...")
                    else:
                        print(f"  R{round_num}: 失败 - {result.error or 'No response'}")

                    time.sleep(0.3)

    return results


def run_multi_turn_test() -> dict:
    """多轮对话测试"""
    results = {}

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"多轮对话测试 - {model}")
        print(f"{'='*60}")

        # 根据模型类型决定是否关闭 thinking
        # 必须 thinking 的模型不关闭，其他模型关闭以获得最低延时
        disable_thinking = model not in THINKING_MODELS_REQUIRED
        mode_str = "thinking OFF" if disable_thinking else "thinking ON (required)"
        print(f"测试模式: {mode_str}")

        conversation_history = []
        model_results = []

        for turn_num, prompt in enumerate(MULTI_TURN_DIALOGUES, 1):
            print(f"\nTurn {turn_num}: {prompt}")

            result_data = test_gemini_streaming(
                model=model,
                prompt=prompt,
                system_instruction=ROLEPLAY_SYSTEM,
                conversation_history=conversation_history,
                disable_thinking=disable_thinking,
                max_tokens=128
            )

            model_results.append({
                "turn": turn_num,
                "prompt": prompt,
                "ttft": result_data["ttft"],
                "total_time": result_data["total_time"],
                "response": result_data["response_text"],
                "error": result_data["error"]
            })

            if result_data["error"] is None and result_data["ttft"] is not None:
                print(f"  TTFT={result_data['ttft']:.3f}s, Total={result_data['total_time']:.3f}s")
                print(f"  回复: {result_data['response_text']}")

                # 更新对话历史
                conversation_history.append({
                    "role": "user",
                    "parts": [{"text": prompt}]
                })
                conversation_history.append({
                    "role": "model",
                    "parts": [{"text": result_data["response_text"]}]
                })
            else:
                print(f"  失败: {result_data['error']}")

            time.sleep(0.3)

        results[model] = model_results

    return results


def generate_report(single_results: List[LatencyResult], multi_results: dict) -> str:
    """生成测试报告"""
    report = []
    report.append("# 日语 Roleplay 对话延时测试报告")
    report.append("")
    report.append(f"**测试场景**: 简单日语日常对话 (Roleplay)")
    report.append(f"**测试轮数**: {TEST_ROUNDS}")
    report.append(f"**System Prompt**: {ROLEPLAY_SYSTEM[:50]}...")
    report.append("")

    # 1. 单轮测试汇总
    report.append("## 1. 单轮对话延时对比")
    report.append("")
    report.append("| 模型 | Thinking | 平均TTFT | 最小TTFT | 最大TTFT | 平均总时间 | 成功率 |")
    report.append("|------|----------|----------|----------|----------|------------|--------|")

    for model in MODELS:
        # 根据模型类型确定测试模式
        if model in THINKING_MODELS_OPTIONAL:
            thinking_modes = [True, False]
        elif model in THINKING_MODELS_REQUIRED:
            thinking_modes = [True]
        else:
            thinking_modes = [False]

        for thinking_enabled in thinking_modes:
            mode_str = "ON" if thinking_enabled else "OFF"
            filtered = [r for r in single_results
                       if r.model == model and r.thinking_enabled == thinking_enabled]
            successful = [r for r in filtered if r.success]

            if successful:
                ttfts = [r.ttft for r in successful if r.ttft]
                totals = [r.total_time for r in successful if r.total_time]

                avg_ttft = statistics.mean(ttfts) if ttfts else 0
                min_ttft = min(ttfts) if ttfts else 0
                max_ttft = max(ttfts) if ttfts else 0
                avg_total = statistics.mean(totals) if totals else 0
                success_rate = len(successful) / len(filtered) * 100 if filtered else 0

                report.append(f"| {model} | {mode_str} | {avg_ttft:.3f}s | {min_ttft:.3f}s | {max_ttft:.3f}s | {avg_total:.3f}s | {success_rate:.0f}% |")
            else:
                report.append(f"| {model} | {mode_str} | N/A | N/A | N/A | N/A | 0% |")

    report.append("")

    # 2. Thinking ON vs OFF 对比 (可选 thinking 的模型)
    report.append("## 2. Thinking 模式对比 (可关闭 thinking 的模型)")
    report.append("")
    report.append("| 模型 | Thinking ON | Thinking OFF | 延时改善 |")
    report.append("|------|-------------|--------------|----------|")

    for model in THINKING_MODELS_OPTIONAL:
        model_results = [r for r in single_results if r.model == model]
        thinking_on = [r for r in model_results if r.thinking_enabled and r.success and r.ttft]
        thinking_off = [r for r in model_results if not r.thinking_enabled and r.success and r.ttft]

        if thinking_on and thinking_off:
            avg_on = statistics.mean([r.ttft for r in thinking_on])
            avg_off = statistics.mean([r.ttft for r in thinking_off])
            improvement = ((avg_on - avg_off) / avg_on) * 100 if avg_on > 0 else 0
            report.append(f"| {model} | {avg_on:.3f}s | {avg_off:.3f}s | {improvement:.1f}% |")
        elif thinking_on:
            avg_on = statistics.mean([r.ttft for r in thinking_on])
            report.append(f"| {model} | {avg_on:.3f}s | N/A | - |")
        elif thinking_off:
            avg_off = statistics.mean([r.ttft for r in thinking_off])
            report.append(f"| {model} | N/A | {avg_off:.3f}s | - |")
        else:
            report.append(f"| {model} | N/A | N/A | - |")

    report.append("")
    report.append("**注意**: gemini-3-pro-preview 和 gemini-2.5-pro 只支持 thinking 模式，无法关闭。")

    report.append("")

    # 3. 多轮对话测试
    report.append("## 3. 多轮对话延时测试")
    report.append("")

    for model, turns in multi_results.items():
        report.append(f"### {model}")
        report.append("")
        report.append("| Turn | 提示 | TTFT | 总时间 | 回复 |")
        report.append("|------|------|------|--------|------|")

        for t in turns:
            ttft_str = f"{t['ttft']:.3f}s" if t['ttft'] else "N/A"
            total_str = f"{t['total_time']:.3f}s" if t['total_time'] else "N/A"
            response_short = (t['response'][:20] + "...") if t['response'] and len(t['response']) > 20 else (t['response'] or "N/A")
            report.append(f"| {t['turn']} | {t['prompt'][:15]}... | {ttft_str} | {total_str} | {response_short} |")

        # 计算平均
        successful_turns = [t for t in turns if t['ttft']]
        if successful_turns:
            avg_ttft = statistics.mean([t['ttft'] for t in successful_turns])
            avg_total = statistics.mean([t['total_time'] for t in successful_turns])
            report.append("")
            report.append(f"**平均 TTFT**: {avg_ttft:.3f}s | **平均总时间**: {avg_total:.3f}s")

        report.append("")

    # 4. 最优配置推荐
    report.append("## 4. 最优配置推荐")
    report.append("")

    # 找出最低 TTFT 的配置
    successful_all = [r for r in single_results if r.success and r.ttft]
    if successful_all:
        best = min(successful_all, key=lambda x: x.ttft)
        report.append(f"**最低单次 TTFT**: {best.ttft:.3f}s")
        report.append(f"- 模型: {best.model}")
        report.append(f"- Thinking: {'ON' if best.thinking_enabled else 'OFF'}")
        report.append("")

        # 按模型分组计算平均
        model_avg = {}
        for model in MODELS:
            model_results = [r for r in successful_all if r.model == model and not r.thinking_enabled]
            if model_results:
                model_avg[model] = statistics.mean([r.ttft for r in model_results])

        if model_avg:
            best_model = min(model_avg, key=model_avg.get)
            report.append(f"**推荐配置 (最低平均延时)**:")
            report.append(f"- 模型: `{best_model}`")
            report.append(f"- Thinking: OFF (`thinkingBudget: 0`)")
            report.append(f"- maxOutputTokens: 128")
            report.append(f"- 平均 TTFT: {model_avg[best_model]:.3f}s")

    report.append("")
    report.append("## 5. 代码配置示例")
    report.append("")
    report.append("```python")
    report.append('payload = {')
    report.append('    "contents": contents,')
    report.append('    "systemInstruction": {')
    report.append('        "parts": [{"text": "あなたは「さくら」という名前の女子大生です..."}]')
    report.append('    },')
    report.append('    "generationConfig": {')
    report.append('        "temperature": 0.8,')
    report.append('        "maxOutputTokens": 128,')
    report.append('        "thinkingConfig": {"thinkingBudget": 0}  # 关闭 thinking')
    report.append('    }')
    report.append('}')
    report.append("```")
    report.append("")

    return "\n".join(report)


def main():
    """主函数"""
    print("=" * 60)
    print("日语 Roleplay 对话延时测试")
    print("=" * 60)

    if not GEMINI_API_KEY:
        print("错误: 请设置 GEMINI_API_KEY 环境变量")
        return

    # 1. 单轮对话测试
    print("\n[1/2] 单轮对话测试...")
    single_results = run_single_turn_tests()

    # 2. 多轮对话测试
    print("\n[2/2] 多轮对话测试...")
    multi_results = run_multi_turn_test()

    # 生成报告
    report = generate_report(single_results, multi_results)

    # 保存报告
    report_file = "/home/andy/work/voice-test/output/japanese_roleplay_latency_report.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    print(report)
    print(f"\n报告已保存到: {report_file}")


if __name__ == "__main__":
    main()
