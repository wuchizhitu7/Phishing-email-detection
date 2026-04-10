import joblib
import pandas as pd
from textblob import TextBlob
import re
from urllib.parse import urlparse
import numpy as np
import email
from email import policy
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# 1. BERT 模型架构定义
class PhishingBertModel(nn.Module):
    def __init__(self, n_numeric_feats=8):
        super(PhishingBertModel, self).__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(768 + n_numeric_feats, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, numeric_feats):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        combined = torch.cat((pooled_output, numeric_feats), dim=1)
        return self.classifier(combined)


# 2. 增强型 EML 预测器
class EMLPredictor:
    def __init__(self, pipeline_path, bert_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

        print(f"正在加载引擎... (设备: {self.device})")

        # 加载 Pipeline 模型
        self.rf_pipeline = joblib.load(pipeline_path)

        # 加载 BERT 引擎
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert_model = PhishingBertModel(n_numeric_feats=8).to(self.device)
        self.bert_model.load_state_dict(torch.load(bert_path, map_location=self.device))
        self.bert_model.eval()

    def _extract_eml_content(self, eml_path):
        """解析 EML 获取清洗后的正文和 URL"""
        with open(eml_path, 'rb') as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        body = ""
        subject = str(msg.get('Subject', ''))

        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype in ["text/plain", "text/html"]:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode(errors='ignore')
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode(errors='ignore')

        soup = BeautifulSoup(body, "html.parser")
        clean_text = soup.get_text()
        urls = re.findall(self.url_pattern, body)

        # 融合标题和正文
        full_content = subject + " " + clean_text
        return full_content, urls

    def _get_numeric_dict(self, body, urls):
        """提取数值特征并返回字典格式，方便转为 DataFrame"""
        blob = TextBlob(body)
        sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if not urls:
            url_feats = [0.0, 0.0, 0, 0, 0.0, 0]
        else:
            lens = [len(u) for u in urls]
            dots = [u.count('.') for u in urls]
            has_at = 1 if any('@' in u for u in urls) else 0
            has_ip = 1 if any(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}', u) for u in urls) else 0
            subdomains = [len(urlparse(u).netloc.split('.')) for u in urls]
            url_feats = [float(np.mean(lens)), float(np.mean(dots)), has_at, has_ip, float(np.mean(subdomains)),
                         len(urls)]

        return {
            'sentiment': sentiment,
            'subjectivity': subjectivity,
            'avg_url_len': url_feats[0],
            'avg_url_dots': url_feats[1],
            'has_at_symbol': url_feats[2],
            'has_ip_url': url_feats[3],
            'avg_subdomains': url_feats[4],
            'url_count': url_feats[5]
        }

    def _get_rf_prob(self, body, num_dict):
        """RF 模型推理"""
        input_data = {'body': body}
        input_data.update(num_dict)
        input_df = pd.DataFrame([input_data])
        return self.rf_pipeline.predict_proba(input_df)[0][1]

    def _get_bert_prob(self, body, num_dict):
        """BERT 模型推理"""
        tokens = self.tokenizer.tokenize(body)
        if len(tokens) > 510:
            tokens = tokens[:255] + tokens[-255:]

        inputs = self.tokenizer.encode_plus(
            tokens, is_split_into_words=True, add_special_tokens=True,
            max_length=512, padding='max_length', truncation=True, return_tensors='pt'
        ).to(self.device)

        num_list = [num_dict[k] for k in ['sentiment', 'subjectivity', 'avg_url_len', 'avg_url_dots',
                                          'has_at_symbol', 'has_ip_url', 'avg_subdomains', 'url_count']]
        num_tensor = torch.tensor([num_list], dtype=torch.float).to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(inputs['input_ids'], inputs['attention_mask'], num_tensor)
            prob = torch.softmax(outputs, dim=1)[0][1].item()
        return prob

    def predict(self, eml_path, mode="BERT", rf_weight=0.4, bert_weight=0.6):
        """
        mode 支持三种模式：
            "RF"       → 仅随机森林
            "BERT"     → 仅 DistilBERT
            "ENSEMBLE" → 加权融合
        """
        body, urls = self._extract_eml_content(eml_path)
        num_dict = self._get_numeric_dict(body, urls)

        if mode == "RF":
            prob = self._get_rf_prob(body, num_dict)
            mode_name = "RF"
        elif mode == "BERT":
            prob = self._get_bert_prob(body, num_dict)
            mode_name = "BERT"
        elif mode == "ENSEMBLE":
            rf_prob = self._get_rf_prob(body, num_dict)
            bert_prob = self._get_bert_prob(body, num_dict)
            prob = rf_weight * rf_prob + bert_weight * bert_prob

            print(f"\n--- [ENSEMBLE 引擎] 检测报告: {eml_path} ---")
            print(f"RF   恶意概率: {rf_prob:.2%}")
            print(f"BERT 恶意概率: {bert_prob:.2%}")
            print(f"Ensemble 恶意概率: {prob:.2%}")
            print(f"判定结果: {'【高风险】钓鱼邮件' if prob > 0.5 else '【安全】正常邮件'}")
            return prob
        else:
            raise ValueError("mode 必须是 'RF'、'BERT' 或 'ENSEMBLE'")

        # 单模型统一报告
        print(f"\n--- [{mode_name} 引擎] 检测报告: {eml_path} ---")
        print(f"恶意概率得分: {prob:.2%}")
        print(f"判定结果: {'【高风险】钓鱼邮件' if prob > 0.5 else '【安全】正常邮件'}")
        return prob


# 3. 运行示例
if __name__ == "__main__":
    predictor = EMLPredictor(
        pipeline_path='phishing_detector_final.pkl',  # 随机森林+TD-IDF
        bert_path='phishing_bert_model.pth'  # BERT 权重
    )

    target_eml = "test.eml"

    print("=" * 60)
    predictor.predict(target_eml, mode="RF")
    print("=" * 60)
    predictor.predict(target_eml, mode="BERT")
    print("=" * 60)
    predictor.predict(target_eml, mode="ENSEMBLE", rf_weight=0.4, bert_weight=0.6)
