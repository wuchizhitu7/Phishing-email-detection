"""
查看 eml 邮件文件的结构和正文内容
使用: python test.py [eml文件路径]
"""
import email
from email import policy
import sys

eml_path = sys.argv[1] if len(sys.argv) > 1 else 'predict_email.eml'

with open(eml_path, 'rb') as f:
    msg = email.message_from_binary_file(f, policy=policy.default)

print(f"From:    {msg.get('From', 'N/A')}")
print(f"To:      {msg.get('To', 'N/A')}")
print(f"Subject: {msg.get('Subject', 'N/A')}")
print(f"Date:    {msg.get('Date', 'N/A')}")
print("=" * 60)

body_parts = []
if msg.is_multipart():
    for part in msg.walk():
        ctype = part.get_content_type()
        cdisp = str(part.get('Content-Disposition', ''))
        if ctype in ('text/plain', 'text/html') and 'attachment' not in cdisp:
            payload = part.get_payload(decode=True)
            if payload:
                body_parts.append((ctype, payload))
else:
    payload = msg.get_payload(decode=True)
    if payload:
        body_parts.append((msg.get_content_type(), payload))

for ctype, data in body_parts:
    print(f"\n--- {ctype} ---")
    text = data.decode('utf-8', errors='replace')
    if ctype == 'text/html':
        from bs4 import BeautifulSoup
        text = BeautifulSoup(text, 'html.parser').get_text()
    print(text[:3000])
    if len(text) > 3000:
        print(f"\n... (截断，总长 {len(text)} 字符)")
