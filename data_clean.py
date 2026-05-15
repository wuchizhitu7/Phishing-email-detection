import re
import unicodedata
import pandas as pd
from bs4 import BeautifulSoup
import os
import email
from email import policy

class EmailProcessor:
    def __init__(self):
        # 预编译正则
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    def _extract_urls(self, text):
        """提取文本中的所有 URL"""
        if not text:
            return []
        return re.findall(self.url_pattern, text)

    def _ultra_clean_text(self, text):
        """处理乱码、转义符、编码异常"""
        if not text:
            return ""
        # 1. 清理HTML标签
        text = BeautifulSoup(text, "html.parser").get_text()
        # 2. 替换基础转义符
        text = text.replace('\xa0', ' ').replace('\t', ' ').replace('\r', ' ')
        # 3. 过滤非打印字符
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch == '\n')
        # 4. 规范化空格
        text = re.sub(r' +', ' ', text)
        # 5. 规范化换行
        text = re.sub(r'\n\s*\n+', '\n', text)
        return text.strip()

def get_trec06c_dataframe(data_dir="trec06c"):
    """解析 TREC06C 中文邮件数据集"""
    processor = EmailProcessor()
    index_path = os.path.join(data_dir, "full", "index")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"TREC06C index not found: {index_path}")

    # 读取标注
    label_map = {}  # filename → label (0=ham, 1=spam)
    with open(index_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                label_str = parts[0].lower()
                file_path = parts[1]
                label_map[file_path] = 0 if label_str == "ham" else 1

    print(f"TREC06C 标注数: {len(label_map)} (ham={sum(1 for v in label_map.values() if v==0)}, spam={sum(1 for v in label_map.values() if v==1)})")

    processed = []
    skipped = 0
    for rel_path, label in label_map.items():
        abs_path = os.path.join(data_dir, rel_path.lstrip("./"))
        if not os.path.isfile(abs_path):
            skipped += 1
            continue

        try:
            with open(abs_path, "rb") as f:
                msg = email.message_from_binary_file(f, policy=policy.default)

            raw_body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    if ctype in ["text/plain", "text/html"]:
                        payload = part.get_payload(decode=True)
                        if payload:
                            raw_body += payload.decode(errors='ignore')
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    raw_body = payload.decode(errors='ignore')

            clean_body = processor._ultra_clean_text(raw_body)
            urls = processor._extract_urls(raw_body)
            subject = str(msg.get("Subject", "Unknown")).strip() or "Unknown"
            sender = str(msg.get("From", "Unknown")).strip() or "Unknown"

            processed.append({
                "subject": subject,
                "sender": sender,
                "reply_to": "None",
                "received_count": 0,
                "body": clean_body if clean_body else "Empty_Body",
                "urls": urls,
                "url_count": len(urls),
                "label": label,
            })
        except Exception:
            skipped += 1

    print(f"成功解析: {len(processed)}, 跳过: {skipped}")
    df = pd.DataFrame(processed)
    df = df.drop_duplicates(subset=["body"], keep="first")
    print(f"去重后: {len(df)}")
    return df


if __name__ == "__main__":
    df = get_trec06c_dataframe()
    print(df.head())