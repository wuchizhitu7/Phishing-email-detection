import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from textblob import TextBlob  
from data_clean import get_cleaned_dataframe

class DeepFeatureExtractor:
    def __init__(self):
        self.phish_keywords = ['verify', 'account', 'update', 'security', 'service', 'login', 'click', 'bank']

    def analyze_text_semantics(self, text):
        """分析正文语义特征"""
        blob = TextBlob(str(text))
        # 1. 情感极性：钓鱼邮件往往情绪波动较大
        sentiment = blob.sentiment.polarity
        # 2. 主观性：钓鱼邮件通常较为主观
        subjectivity = blob.sentiment.subjectivity
        return sentiment, subjectivity

    def analyze_url_structure(self, url_list):
        """分析URL基本结构特征"""
        if not url_list or len(url_list) == 0:
            return [0] * 6  # 如果没URL，特征全为0

        # 统计特征
        lengths = []
        dot_counts = []
        has_at = 0
        has_ip = 0
        subdomain_counts = []

        for url in url_list:
            lengths.append(len(url))
            dot_counts.append(url.count('.'))
            if '@' in url: has_at = 1
            # 正则判断是否包含IP地址
            if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url): has_ip = 1

            # 解析域名
            domain = urlparse(url).netloc
            subdomain_counts.append(len(domain.split('.')))

        # 返回平均值作为该邮件的特征
        return [
            np.mean(lengths),
            np.mean(dot_counts),
            has_at,
            has_ip,
            np.mean(subdomain_counts),
            len(url_list)  # 原始计数
        ]


if __name__ == "__main__":
    # 1. 获取数据
    print("正在加载并清洗原始数据...")
    df = get_cleaned_dataframe()

    # 2. 初始化提取器
    extractor = DeepFeatureExtractor()

    # 3. 提取语义特征
    print("正在分析语义特征...")
    df['sentiment'], df['subjectivity'] = zip(*df['body'].apply(extractor.analyze_text_semantics))

    # 4. 提取 URL 结构特征
    print("正在分析 URL 结构...")
    url_features = df['urls'].apply(extractor.analyze_url_structure)
    feature_cols = ['avg_url_len', 'avg_url_dots', 'has_at_symbol', 'has_ip_url', 'avg_subdomains', 'url_count_check']
    df[feature_cols] = pd.DataFrame(url_features.tolist(), index=df.index)

    # 5. 保存带有深度特征的新数据集
    df.to_csv("enriched_emails_dataset.csv", index=False)
    print("深度特征提取完成！结果已保存至 enriched_emails_dataset.csv")
