import joblib
import pandas as pd
from textblob import TextBlob
import re
from urllib.parse import urlparse
import numpy as np
import email
from email import policy
from bs4 import BeautifulSoup

# 加载模型
model = joblib.load('phishing_detector_final.pkl')

class EMLPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    def _extract_eml_content(self, eml_path):
        """解析 EML 文件，提取正文和 URL"""
        with open(eml_path, 'rb') as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        # 提取正文
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                cdisp = str(part.get("Content-Disposition"))
                if ctype in ["text/plain", "text/html"] and "attachment" not in cdisp:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode(errors='ignore')
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode(errors='ignore')

        # 清洗 HTML 并提取 URL
        soup = BeautifulSoup(body, "html.parser")
        clean_text = soup.get_text()
        urls = re.findall(self.url_pattern, body)

        return clean_text, urls

    def predict(self, eml_path):
        """对单个 EML 文件进行预测"""
        body, urls = self._extract_eml_content(eml_path)

        # 1. 语义特征
        blob = TextBlob(body)
        sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # 2. URL 结构特征
        if not urls:
            url_feats = [0, 0, 0, 0, 0, 0]
        else:
            lens = [len(u) for u in urls]
            dots = [u.count('.') for u in urls]
            has_at = 1 if any('@' in u for u in urls) else 0
            has_ip = 1 if any(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}', u) for u in urls) else 0
            subdomains = [len(urlparse(u).netloc.split('.')) for u in urls]
            url_feats = [np.mean(lens), np.mean(dots), has_at, has_ip, np.mean(subdomains), len(urls)]

        # 3. 构造输入 DataFrame
        input_df = pd.DataFrame([{
            'body': body,
            'sentiment': sentiment,
            'subjectivity': subjectivity,
            'avg_url_len': url_feats[0],
            'avg_url_dots': url_feats[1],
            'has_at_symbol': url_feats[2],
            'has_ip_url': url_feats[3],
            'avg_subdomains': url_feats[4],
            'url_count': url_feats[5]
        }])

        # 4. 执行预测
        prob = self.model.predict_proba(input_df)[0][1]
        label = self.model.predict(input_df)[0]

        print(f"\n--- 邮件检测报告: {eml_path} ---")
        print(f"正文长度: {len(body)} 字符")
        print(f"提取链接数: {len(urls)}")
        print(f"恶意概率得分: {prob:.2%}")
        print(f"判定结果: {'【高风险】钓鱼邮件' if label == 1 else '【安全】正常邮件'}")

        if label == 1:
            print("风险提示: 该邮件包含典型的钓鱼特征，请勿点击文中链接或输入个人信息。")

if __name__ == "__main__":
    predictor = EMLPredictor('phishing_detector_final.pkl')
    predictor.predict(r"D:\下载\AD_您有机会赢取丰厚大奖：奖品总价值_¥3,500,000.eml")
