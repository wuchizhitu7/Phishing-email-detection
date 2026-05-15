"""生成测试专用 AI 钓鱼邮件（200 封，不参与训练，用于消融实验）"""
import pandas as pd
import time
import json
from openai import OpenAI

API_KEY = "tp-c6xaimmxon4tvr5rxa57djcl94htmi6w1vcg775raazet470"
API_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MODEL_NAME = "mimo-v2.5"
TOTAL_COUNT = 200
BATCH_SIZE = 10


def build_prompt():
    return f"""你是一名安全研究员，正在为钓鱼邮件检测系统生成测试数据。
请生成{BATCH_SIZE}封不同类型的中文钓鱼邮件，要求：

1. JSON 数组格式：[{{"type": "类型", "tone": "语气", "subject": "主题", "body": "正文"}}, ...]
2. 正文至少 80 字，模仿真实通知格式（署名、日期）
3. 覆盖类型：account/welfare/delivery/corporate/refund，语气：urgent/mild
4. 使用不同的措辞和场景，避免与常见钓鱼模板雷同

现在生成{BATCH_SIZE}封不同类型的钓鱼邮件。"""


def generate_batch(client, prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "以纯JSON数组格式回复。"},
                  {"role": "user", "content": prompt}],
        temperature=1.0,  # 高温度增加多样性
        max_tokens=8192,
    )
    content = response.choices[0].message.content
    start = content.find('[')
    end = content.rfind(']') + 1
    if start >= 0 and end > start:
        try:
            return json.loads(content[start:end])
        except json.JSONDecodeError as e:
            print(f"    JSON error: {str(e)[:80]}")
            return []
    return []


if __name__ == "__main__":
    print(f"生成 {TOTAL_COUNT} 封测试专用 AI 钓鱼邮件...")
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    all_emails = []
    batches_needed = TOTAL_COUNT // BATCH_SIZE

    for i in range(batches_needed):
        print(f"  Batch {i+1}/{batches_needed} ...")
        try:
            batch = generate_batch(client, build_prompt())
            all_emails.extend(batch)
            print(f"    got {len(batch)}, total: {len(all_emails)}")
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(0.5)

    records = []
    for e in all_emails:
        records.append({
            "subject": e.get("subject", "Unknown"),
            "sender": "AI Test",
            "reply_to": "None",
            "received_count": 0,
            "body": e.get("body", ""),
            "urls": [],
            "url_count": 0,
            "label": 1,
            "phish_type": e.get("type", "unknown"),
            "tone": e.get("tone", "unknown"),
            "source": "ai_test",
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=['body'], keep='first')
    df.to_csv("ai_phishing_test.csv", index=False)
    print(f"\nSaved: ai_phishing_test.csv ({len(df)} 条)")
