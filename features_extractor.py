import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from snownlp import SnowNLP
from data_clean import get_trec06c_dataframe
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


class DeepFeatureExtractor:
    def __init__(self):
        self.phish_keywords = [
            '验证', '账号', '账户', '安全', '登录', '点击', '链接', '银行',
            '冻结', '停用', '更新', '确认', '密码', '身份', '验证码',
            '退款', '中奖', '补贴', '奖金', '领取', '福利', '优惠',
            '紧急', '立即', '尽快', '否则', '关闭', '异常', '风险',
            '客服', '通知', '系统', '升级', '激活', '认证'
        ]
        self.ppl_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ppl_tokenizer = None
        self.ppl_model = None

    def analyze_text_semantics(self, text):
        """中文语义特征：SnowNLP 情感 + 钓鱼关键词密度"""
        text = str(text)
        s = SnowNLP(text)
        sentiment = s.sentiments
        kw_count = sum(1 for kw in self.phish_keywords if kw in text)
        keyword_density = kw_count / max(len(text), 1)
        return sentiment, keyword_density

    def compute_perplexity(self, texts, batch_size=4):
        """用 MLM 模型计算困惑度。AI 文本偏低，人类文本偏高"""
        if self.ppl_model is None:
            print("  加载 MLM 模型用于 perplexity...")
            self.ppl_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            self.ppl_model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
            self.ppl_model.to(self.ppl_device)
            self.ppl_model.eval()

        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t)[:256] for t in texts[i:i + batch_size]]
            enc = self.ppl_tokenizer(
                batch_texts, return_tensors='pt', padding=True,
                truncation=True, max_length=256
            ).to(self.ppl_device)
            with torch.no_grad():
                outputs = self.ppl_model(**enc, labels=enc['input_ids'])
                loss = outputs.loss
            ppl = torch.exp(loss).cpu().item()
            results.extend([ppl] * len(batch_texts))
        return np.array(results)

    def analyze_url_structure(self, url_list):
        """URL 结构特征"""
        if not url_list or len(url_list) == 0:
            return [0] * 6

        lengths, dot_counts, subdomain_counts = [], [], []
        has_at, has_ip = 0, 0

        for url in url_list:
            lengths.append(len(url))
            dot_counts.append(url.count('.'))
            if '@' in url:
                has_at = 1
            if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
                has_ip = 1
            try:
                domain = urlparse(url).netloc
                subdomain_counts.append(len(domain.split('.')))
            except ValueError:
                subdomain_counts.append(0)

        return [
            np.mean(lengths), np.mean(dot_counts),
            has_at, has_ip,
            np.mean(subdomain_counts),
            len(url_list)
        ]


if __name__ == "__main__":
    # 加载 TREC06C（仅保留 ham）
    print("正在加载 TREC06C 中文邮件数据集...")
    df_tr = get_trec06c_dataframe("data_trec06c/trec06c")
    df_tr = df_tr[df_tr['label'] == 0]

    # 加载 CHIFRAUD
    print("正在加载 CHIFRAUD 数据集（各采样 3 万）...")
    df_ch = pd.read_csv("chifraud_emails.csv")
    df_ch_normal = df_ch[df_ch['label'] == 0].sample(n=30000, random_state=42)
    df_ch_fraud = df_ch[df_ch['label'] == 1].sample(n=30000, random_state=42)
    df_ch = pd.concat([df_ch_normal, df_ch_fraud], ignore_index=True)
    print(f"  CHIFRAUD 采样后: {len(df_ch)} (正常: {len(df_ch_normal)}, 欺诈: {len(df_ch_fraud)})")

    # 加载 AI 生成钓鱼邮件
    import os as _os
    ai_path = "ai_phishing_emails.csv"
    frames = [df_tr, df_ch]
    if _os.path.exists(ai_path):
        print("正在加载 AI 生成钓鱼邮件...")
        df_ai = pd.read_csv(ai_path)
        print(f"  AI 钓鱼: {len(df_ai)} 条")
        frames.append(df_ai)

    df = pd.concat(frames, ignore_index=True)
    print(f"数据总量: {len(df)}, label=0: {(df['label']==0).sum()}, label=1: {(df['label']==1).sum()}")

    extractor = DeepFeatureExtractor()

    print("正在分析语义特征...")
    df['sentiment'], df['subjectivity'] = zip(*df['body'].apply(extractor.analyze_text_semantics))

    print("正在计算困惑度 (perplexity)...")
    df['perplexity'] = extractor.compute_perplexity(df['body'].tolist())
    df['perplexity'] = np.log1p(df['perplexity'])  # 取对数压缩到 ~10~12 范围，避免 FP16 NaN

    print("正在分析 URL 结构...")
    url_features = df['urls'].apply(extractor.analyze_url_structure)
    feature_cols = ['avg_url_len', 'avg_url_dots', 'has_at_symbol', 'has_ip_url', 'avg_subdomains', 'url_count']
    df[feature_cols] = pd.DataFrame(url_features.tolist(), index=df.index)

    df.to_csv("enriched_emails_dataset.csv", index=False)
    print(f"特征提取完成 → enriched_emails_dataset.csv ({len(df)} 条)")
