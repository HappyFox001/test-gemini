# Gemini 模型对比测试

对比测试 Gemini 模型在响应延时和情感交互质量上的表现。

## 快速开始

```bash
# 1. 安装依赖
pip3 install google-genai

# 2. 设置 API Key
export GEMINI_API_KEY='your-api-key-here'

# 3. 运行测试
python3 test_gemini_models.py

# 或使用脚本
./run_test.sh
```

## 获取 API Key

https://aistudio.google.com/app/apikey

## 查看可用模型

```bash
export GEMINI_API_KEY='your-api-key-here'
python3 list_models.py
```

## 配置

编辑 `test_gemini_models.py` 文件：

- **MODELS**: 修改要测试的模型列表
- **REQUEST_GAP**: 修改请求间隔时间（秒）
- **TEST_PROMPT**: 修改测试提示词

## 输出

- 实时显示每个模型的响应时间和回复内容
- 总结报告显示速度对比
- 结果自动保存到 `test_results_*.json` 文件
