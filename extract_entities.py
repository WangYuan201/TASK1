# extract_entities.py
import os
import json
import re
import pdfplumber
from openai import OpenAI

# ================= 1. 配置区 =================
PDF_PATH = "A case of portal vein recanalization and symptomatic heart failure.pdf"
OUTPUT_JSON = "output.json"

# 检查 API Key 是否已配置
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("未找到 DASHSCOPE_API_KEY 环境变量，请先配置！")

# 初始化百炼客户端 (兼容 OpenAI SDK)
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# ================= 2. 提取 PDF 文本 =================
def extract_text_from_pdf(pdf_path):
    print(f"正在读取 PDF: {pdf_path} ...")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            print(f"  - 已解析第 {i + 1} 页")
    return text.strip()


# ================= 3. 设计 Prompt =================
# 注意：JSON 示例中的单大括号 {} 必须写成双大括号 {{}}，否则 .format() 会报错
PROMPT_TEMPLATE = """你是一名资深的医学信息抽取专家。请阅读以下病例报告，并从中提取关键的医学实体。

严格按照以下 JSON 结构输出，不要输出任何多余的解释文字，不要使用 Markdown 代码块包裹。
如果某项信息在文本中未提及，请填 null（对象型）或 []（数组型）。

JSON 结构如下：
{{
  "patient_basic_info": {{
    "name": null,
    "age": null,
    "gender": null,
    "occupation": null
  }},
  "chief_complaints": [],
  "symptoms": [],
  "past_history": [],
  "diagnosis": [],
  "treatment": []
}}

病例原文如下：
{case_text}
"""


# ================= 4. 调用大模型 API =================
def call_llm(prompt):
    print("正在调用大模型 API (Qwen) 进行抽取，请稍候...")
    try:
        response = client.chat.completions.create(
            model="qwen-plus",  # 如果你用的是 qwen-turbo，请替换这里
            messages=[
                {"role": "system", "content": "你是一个严谨的医学实体抽取助手，只输出 JSON 格式数据。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # 设为 0 保证结果稳定
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API 调用失败: {e}")
        return None


# ================= 5. 解析与清洗 JSON =================
def parse_json_output(raw_text):
    print("正在解析模型返回结果...")
    if not raw_text:
        return None

    # 清洗：去掉可能存在的 ```json 和 ``` 包裹
    cleaned_text = re.sub(r"```json\s*", "", raw_text)
    cleaned_text = re.sub(r"```\s*", "", cleaned_text).strip()

    # 容错：截取第一个 { 到最后一个 } 之间的内容
    start_idx = cleaned_text.find("{")
    end_idx = cleaned_text.rfind("}")

    if start_idx != -1 and end_idx != -1:
        cleaned_text = cleaned_text[start_idx:end_idx + 1]

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")
        print("原始返回内容如下：\n", raw_text)
        return None


# ================= 6. 主流程 =================
def main():
    # 1. 抽文字
    case_text = extract_text_from_pdf(PDF_PATH)
    if not case_text:
        print("警告：PDF 提取文字为空，请检查 PDF 是否为扫描件。")
        return

    # 2. 拼 Prompt
    prompt = PROMPT_TEMPLATE.format(case_text=case_text)

    # 3. 调 API
    raw_response = call_llm(prompt)
    if not raw_response:
        return

    # 4. 解析 JSON
    result_data = parse_json_output(raw_response)

    # 5. 保存结果
    if result_data:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)
        print(f"✅ 抽取成功！结果已保存至 {OUTPUT_JSON}")
    else:
        print("❌ 抽取失败，未能生成有效的 JSON 文件。")


if __name__ == "__main__":
    main()