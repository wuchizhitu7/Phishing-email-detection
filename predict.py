import joblib
import pandas as pd
import re
from urllib.parse import urlparse
import numpy as np
import email
from email import policy
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tokenizer_utils import jieba_tokenizer
from url_security import analyze_urls, format_url_report
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from torch import no_grad, exp as torch_exp, tensor

MODEL_NAME = "bert-base-chinese"

MODEL_NAME = "bert-base-chinese"

# 1. BERT 模型架构定义
class PhishingBertModel(nn.Module):
    def __init__(self, n_numeric_feats=8):
        super(PhishingBertModel, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
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

        # 加载Pipeline模型
        self.rf_pipeline = joblib.load(pipeline_path)

        # 加载BERT引擎
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.bert_model = PhishingBertModel(n_numeric_feats=9).to(self.device)
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
        """提取数值特征"""
        from snownlp import SnowNLP
        s = SnowNLP(body)
        sentiment = s.sentiments
        cn_keywords = ['验证', '账号', '银行', '冻结', '中奖', '补贴', '点击', '链接', '密码']
        kw_count = sum(1 for kw in cn_keywords if kw in body)
        subjectivity = kw_count / max(len(body), 1)

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

        # Perplexity
        tokenizer_ppl = AutoTokenizer.from_pretrained(MODEL_NAME)
        model_ppl = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(self.device)
        model_ppl.eval()
        enc = tokenizer_ppl(str(body)[:256], return_tensors='pt', truncation=True,
                            max_length=256, padding=True).to(self.device)
        with no_grad():
            outputs = model_ppl(**enc, labels=enc['input_ids'])
            perplexity = np.log1p(torch_exp(outputs.loss).item())  # 与训练一致取 log

        return {
            'sentiment': sentiment,
            'subjectivity': subjectivity,
            'avg_url_len': url_feats[0],
            'avg_url_dots': url_feats[1],
            'has_at_symbol': url_feats[2],
            'has_ip_url': url_feats[3],
            'avg_subdomains': url_feats[4],
            'url_count': url_feats[5],
            'perplexity': perplexity,
        }

    def _get_rf_prob(self, body, num_dict):
        """RF模型推理"""
        input_data = {'body': body}
        input_data.update(num_dict)
        input_df = pd.DataFrame([input_data])
        return self.rf_pipeline.predict_proba(input_df)[0][1]

    def _get_bert_prob(self, body, num_dict):
        """BERT模型推理"""
        inputs = self.tokenizer.encode_plus(
            body, add_special_tokens=True,
            max_length=256, padding='max_length', truncation=True, return_tensors='pt'
        ).to(self.device)

        num_list = [num_dict[k] for k in ['sentiment', 'subjectivity', 'avg_url_len', 'avg_url_dots',
                                          'has_at_symbol', 'has_ip_url', 'avg_subdomains', 'url_count',
                                          'perplexity']]
        num_tensor = torch.tensor([num_list], dtype=torch.float).to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(inputs['input_ids'], inputs['attention_mask'], num_tensor)
            prob = torch.softmax(outputs, dim=1)[0][1].item()
        return prob

    def predict(self, eml_path, mode="BERT", rf_weight=0.4, bert_weight=0.6):
        """
        mode支持三种模式：
            "RF"       → 仅随机森林
            "BERT"     → 仅 DistilBERT
            "ENSEMBLE" → 加权融合
        """
        body, urls = self._extract_eml_content(eml_path)
        num_dict = self._get_numeric_dict(body, urls)

        # URL 安全分析 + 风险偏置
        url_results = analyze_urls(urls)
        url_report = format_url_report(url_results)
        url_max_risk = max((r['risk_score'] for r in url_results), default=0)
        if url_max_risk >= 6:
            url_boost = 0.15
        elif url_max_risk >= 4:
            url_boost = 0.08
        else:
            url_boost = 0.0

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

            if url_boost > 0:
                prob = min(prob + url_boost, 1.0)

            print(f"\n--- [ENSEMBLE 引擎] 检测报告: {eml_path} ---")
            print(f"RF   恶意概率: {rf_prob:.2%}")
            print(f"BERT 恶意概率: {bert_prob:.2%}")
            print(f"Ensemble 恶意概率: {prob:.2%}")
            if url_boost > 0:
                print(f"URL 安全偏置: +{url_boost:.0%} (最大风险分 {url_max_risk})")
            print(f"判定结果: {'【高风险】钓鱼邮件' if prob > 0.5 else '【安全】正常邮件'}")
            print(f"\n--- URL 安全分析 ---")
            print(url_report)
            return prob
        else:
            raise ValueError("mode 必须是 'RF'、'BERT' 或 'ENSEMBLE'")

        # 单模型: URL 偏置
        prob = min(prob + url_boost, 1.0)

        print(f"\n--- [{mode_name} 引擎] 检测报告: {eml_path} ---")
        print(f"恶意概率得分: {prob:.2%}")
        if url_boost > 0:
            print(f"URL 安全偏置: +{url_boost:.0%} (最大风险分 {url_max_risk})")
        print(f"判定结果: {'【高风险】钓鱼邮件' if prob > 0.5 else '【安全】正常邮件'}")
        print(f"\n--- URL 安全分析 ---")
        print(url_report)
        return prob


if __name__ == "__main__":
    predictor = EMLPredictor(
        pipeline_path='phishing_detector_final.pkl',
        bert_path='phishing_bert_model.pth'
    )

    target_eml = r"predict_email.eml"

    print("=" * 60)
    predictor.predict(target_eml, mode="RF")
    print("=" * 60)
    predictor.predict(target_eml, mode="BERT")
    print("=" * 60)
    predictor.predict(target_eml, mode="ENSEMBLE", rf_weight=0.4, bert_weight=0.6)