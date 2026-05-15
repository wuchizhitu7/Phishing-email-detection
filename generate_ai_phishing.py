import pandas as pd
import time
import json
from openai import OpenAI

API_KEY = "tp-c6xaimmxon4tvr5rxa57djcl94htmi6w1vcg775raazet470"
API_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MODEL_NAME = "mimo-v2.5"

TOTAL_COUNT = 5000       # 总生成数量
BATCH_SIZE = 10          # 每次 API 调用生成数量

PHISHING_TYPES = {
    "account": "账号安全类",
    "welfare": "福利补贴类",
    "delivery": "快递物流类",
    "corporate": "企业通知冒名类",
    "refund": "退款理赔类",
}

TONES = {
    "urgent": "紧急胁迫语气：如账号将在X小时内停用、资金将被冻结",
    "mild": "温和商务语气：如温馨提示、请及时处理，无胁迫感",
}

TEMPLATES = {
    "account": [
        "【{company}】您的账户出现异常登录，为保障您的资金安全，请于{time}小时内点击下方链接验证身份，逾期账户将被冻结。",
        "尊敬的{app_name}用户，系统检测到您的账号于{location}异地登录。如非本人操作，请立即点击链接修改密码：",
        "您的{service}账户安全等级过低，根据监管要求需在{time}日前完成实名升级，请点击链接认证。",
        "{service_name}安全中心提醒您：您有一笔来自{location}的异常转账请求，如非本人操作请立即登录验证取消。",
    ],
    "welfare": [
        "尊敬的员工：{company_name}年度员工福利补贴已发放，请于{date}前登录企业服务平台确认领取。详情请点击链接查看。",
        "恭喜您！您已被选为{platform}平台{activity}活动幸运用户，获得{amount}元{prize}领取资格，点击链接填写领取信息。",
        "【{gov_dept}】关于{year}年度{policy_name}专项补贴发放的通知：您有一笔{amount}元补贴待领取，请登录官网确认。",
        "尊敬的{title}：根据{company}年度福利计划，您已获得{amount}元的{card_type}礼品卡，请于{date}前激活使用。",
    ],
    "delivery": [
        "【{courier}】您的快递{tracking}因{reason}无法派送，请于{time}小时内补充{info_type}信息，点击链接处理。",
        "您好，您的包裹{tracking}在清关时被海关抽查，需缴纳清关费用{amount}元。请点击链接完成支付，以免延误。",
        "【物流通知】您的国际快递{tracking}因地址不完整滞留，请点击下方链接补充详细地址。超时将退回寄件方。",
        "{courier}快递提醒：您的{package_type}已到达{location}中转站，因未能联系收件人，请点击链接重新预约派送。",
    ],
    "corporate": [
        "各位同事：接{dept}通知，公司将于{date}进行{system}系统安全升级。请于截止日期前点击链接完成新系统身份验证，否则将影响正常办公。",
        "【{company}人事部】关于{year}年度员工{task}的通知：请于{date}前登录内部系统完成信息核对，未完成将影响薪资发放。",
        "尊敬的{position}：IT部门检测到您的企业邮箱存在安全漏洞，需立即更新安全证书。请点击链接执行修复程序。|此邮件由{company}自动化系统发出，如有疑问请联系IT支持。",
        "【紧急会议通知】{leader}总将于{date}召开{dept}部门安全合规会议，请提前点击链接下载会议资料并确认参会。",
    ],
    "refund": [
        "【{platform}客服】关于您订单{order_id}的退款申请已受理，退款金额{amount}元将退回您的支付账户。请点击链接查看退款进度。",
        "尊敬的{platform}用户：您购买的{product}因质量问题需召回，请于{date}前点击链接填写退款/换货信息，我们将尽快为您处理。",
        "【退款通知】您于{date}在{platform}购买的{product}因{reason}无法发货，已为您办理全额退款。点击链接确认退款到账情况。",
        "{insurance_type}理赔进度通知：您的案件{case_id}已审核通过，理赔金额{amount}元。请点击链接提供收款账户信息以完成赔付。",
    ],
}


def build_prompt():
    prompt = f"""你是一名安全研究员，正在为钓鱼邮件检测系统生成训练数据。
请生成{BATCH_SIZE}封不同类型的中文钓鱼邮件，要求：

1. 严格使用 JSON 数组格式输出：[{{"type": "类型", "tone": "语气", "subject": "主题", "body": "正文"}}, ...]
2. 邮件正文至少 80 字，语气自然，模仿真实商务通知格式（含发件人署名、日期等细节）
3. 覆盖以下类型和语气：
   - 类型：账号安全(account)、福利补贴(welfare)、快递物流(delivery)、企业通知(corporate)、退款理赔(refund)
   - 语气：紧急胁迫(urgent)、温和商务(mild)
4. 不要出现明显拼写错误或乱码
5. 邮件看起来像真实的商业通知，不要过于明显是诈骗

示例：
[
  {{
    "type": "account",
    "tone": "urgent",
    "subject": "【安全通知】您的Apple ID检测到异常登录",
    "body": "尊敬的Apple用户：\\n\\n系统检测到您的Apple ID于10分钟前在未知设备（iPhone 14 Pro，IP: 103.xx.xx.xx）登录。\\n\\n如非本人操作，您的账户可能已被盗用。为保障您的资金和隐私安全，请立即点击下方链接验证并修改密码：\\n\\nhttps://appleid-verify.safety-check.com\\n\\n若您在24小时内未完成验证，账户将被临时冻结以保护您的数据。\\n\\nApple安全中心\\n2024年1月15日"
  }}
]

现在生成{BATCH_SIZE}封不同类型的钓鱼邮件。"""
    return prompt


def generate_batch(client, prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "你是一个网络安全研究员，帮助生成钓鱼邮件检测的训练数据。以纯JSON数组格式回复，不要包含任何解释文字。"},
                  {"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=8192,
    )
    content = response.choices[0].message.content
    start = content.find('[')
    end = content.rfind(']') + 1
    if start >= 0 and end > start:
        try:
            return json.loads(content[start:end])
        except json.JSONDecodeError as e:
            print(f"    JSON error at char {e.pos}: {str(e)[:100]}")
            return []
    print(f"    No JSON array found (len={len(content)})")
    return []


if __name__ == "__main__":
    print(f"API: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    all_emails = []
    batches_needed = TOTAL_COUNT // BATCH_SIZE

    for i in range(batches_needed):
        print(f"  Batch {i+1}/{batches_needed} ...")
        prompt = build_prompt()
        try:
            batch = generate_batch(client, prompt)
            all_emails.extend(batch)
            print(f"    got {len(batch)} emails, total: {len(all_emails)}")
        except Exception as e:
            print(f"    ERROR (skipping): {e}")
            time.sleep(3)


    # 转成 DataFrame
    records = []
    for e in all_emails:
        records.append({
            "subject": e.get("subject", "Unknown"),
            "sender": "AI Generated",
            "reply_to": "None",
            "received_count": 0,
            "body": e.get("body", ""),
            "urls": [],
            "url_count": 0,
            "label": 1,  # 钓鱼邮件
            "phish_type": e.get("type", "unknown"),
            "tone": e.get("tone", "unknown"),
            "source": "ai_generated",
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=['body'], keep='first')
    print(f"\nGenerated: {len(df)} unique emails after dedup")

    out = "ai_phishing_emails.csv"
    df.to_csv(out, index=False)
    print(f"Saved to {out}")
