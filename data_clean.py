import mailbox
import re
import unicodedata
import pandas as pd
from bs4 import BeautifulSoup


class EmailProcessor:
    def __init__(self):
        # 预编译正则提高处理效率
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    def _extract_urls(self, text):
        """提取文本中的所有 URL"""
        if not text:
            return []
        return re.findall(self.url_pattern, text)

    def _get_raw_body(self, message):
        """递归获取邮件原始正文"""
        body = ""
        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                cdisp = str(part.get("Content-Disposition"))
                if content_type in ["text/plain", "text/html"] and "attachment" not in cdisp:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode(errors='ignore')
        else:
            payload = message.get_payload(decode=True)
            if payload:
                body = payload.decode(errors='ignore')
        return body

    def _ultra_clean_text(self, text):
        """深度清洗：处理乱码、转义符、编码异常"""
        if not text:
            return ""
        # 1. 清理 HTML 标签
        text = BeautifulSoup(text, "html.parser").get_text()
        # 2. 替换基础转义符
        text = text.replace('\xa0', ' ').replace('\t', ' ').replace('\r', ' ')
        # 3. 过滤非打印字符（处理编码导致的 '?' 乱码）
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch == '\n')
        # 4. 规范化空格
        text = re.sub(r' +', ' ', text)
        # 5. 规范化换行（保留一个换行，压缩连续空行）
        text = re.sub(r'\n\s*\n+', '\n', text)
        return text.strip()

    def process_mbox(self, file_path, label):
        """解析 mbox 文件并返回结构化数据列表"""
        mbox = mailbox.mbox(file_path)
        processed_emails = []

        for message in mbox:
            # --- 维度 1: Header 处理 ---
            header_info = {
                "subject": str(message['Subject']).strip() if message['Subject'] else "Unknown",
                "from": str(message['From']).strip() if message['From'] else "Unknown",
                "date": str(message['Date']).strip() if message['Date'] else "Unknown",
                "reply_to": str(message['Reply-To']).strip() if message['Reply-To'] else "None",
                "received_count": len(message.get_all('Received') or [])
            }

            # --- 维度 2: Body 处理 ---
            raw_body = self._get_raw_body(message)
            clean_body = self._ultra_clean_text(raw_body)

            # --- 维度 3: URL 处理 ---
            # 注意：从原始 body 提取 URL 避免清洗过程破坏链接结构
            urls = self._extract_urls(raw_body)

            # 整合数据
            processed_emails.append({
                "subject": header_info["subject"],
                "sender": header_info["from"],
                "reply_to": header_info["reply_to"],
                "received_count": header_info["received_count"],
                "body": clean_body if clean_body else "Empty_Body",
                "urls": urls,
                "url_count": len(urls),
                "label": label  # 0 为正常, 1 为钓鱼
            })

        return processed_emails


# ... 前面的类定义保持不变 ...

def get_cleaned_dataframe():
    """封装解析逻辑，供外部调用"""
    processor = EmailProcessor()
    # 这里的路径请确保正确
    normal_data = processor.process_mbox("emails-enron.mbox", label=0)
    phishing_data = processor.process_mbox("emails-phishing.mbox", label=1)

    full_data = normal_data + phishing_data
    df = pd.DataFrame(full_data)
    # 去重逻辑
    df = df.drop_duplicates(subset=['body'], keep='first')
    return df


if __name__ == "__main__":
    # 只有直接运行此脚本时才会执行
    df = get_cleaned_dataframe()
    print(df.head())