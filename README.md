# Gemini 模型延时性能测试

测试 Gemini 3.0 Flash 和 3.0 Pro 的延时性能，包括：
- 首字延时（TTFB）
- 总响应时间
- 对话间隔
- 响应速度（字符/秒）

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

## 测试内容

- 两轮对话测试
- 记录每轮的首字延时和总时间
- 计算总响应长度和总时间
- 输出简洁的对比表格

## 配置

编辑 `test_gemini_models.py` 文件：

- **MODELS**: 测试的模型（默认：3-flash 和 3-pro）
- **PROMPTS**: 两轮对话的提示词
- **CONVERSATION_GAP**: 对话间隔时间（默认 0.5 秒）

## 输出

### 控制台输出
```
指标                 gemini-3-flash       gemini-3-pro
--------------------------------------------------------------------------------
[第1轮] 首字延时       0.245秒             0.312秒
[第1轮] 总时间         1.234秒             1.567秒
[第1轮] 响应长度       156字符              178字符
对话间隔              0.501秒             0.501秒
[第2轮] 首字延时       0.198秒             0.287秒
[第2轮] 总时间         1.123秒             1.445秒
[第2轮] 响应长度       142字符              165字符
--------------------------------------------------------------------------------
总响应长度            298字符              343字符
总响应时间            2.357秒             3.012秒
平均速度              126.4字符/秒          113.9字符/秒
```

### JSON 文件
结果保存到 `performance_test_*.json`，只包含时间数据：
- 首字延时
- 总响应时间
- 响应长度
- 对话间隔
