import gradio as gr
import torch
import torch.nn as nn
import joblib
import re
import numpy as np
import pandas as pd
from textblob import TextBlob
from urllib.parse import urlparse
from transformers import AutoTokenizer, AutoModel
import email
from email import policy

# 1. 参数与路径配置
DEVICE = torch.cuda.is_available() and "cuda" or "cpu"
MODEL_NAME = "distilbert-base-uncased"
RF_MODEL_PATH = 'phishing_detector_final.pkl'
BERT_MODEL_PATH = 'phishing_bert_model.pth'

NUMERIC_COLS = [
    'sentiment', 'subjectivity', 'avg_url_len', 'avg_url_dots',
    'has_at_symbol', 'has_ip_url', 'avg_subdomains', 'url_count'
]


# 2. 重新定义 BERT 模型架构
class PhishingBertModel(nn.Module):
    def __init__(self, n_numeric_feats):
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


# 3.特征提取与模型加载
class InferenceEngine:
    def __init__(self):
        # 加载 RF
        self.rf_pipeline = joblib.load(RF_MODEL_PATH)

        # 加载 BERT
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.bert_model = PhishingBertModel(len(NUMERIC_COLS)).to(DEVICE)
        self.bert_model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=DEVICE))
        self.bert_model.eval()

    def extract_eml_data(self, file_path):
        """解析 eml 文件并计算特征"""
        with open(file_path, 'rb') as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        # 提取正文
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode(errors='ignore')
                    break
        else:
            body = msg.get_payload(decode=True).decode(errors='ignore')

        if not body: body = "Empty Body"

        # 提取 URL
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, body)

        # 计算数值特征
        blob = TextBlob(body)
        sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if not urls:
            url_feats = [0] * 6
        else:
            lens = [len(u) for u in urls]
            dots = [u.count('.') for u in urls]
            has_at = 1 if any('@' in u for u in urls) else 0
            has_ip = 1 if any(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}', u) for u in urls) else 0
            subdomains = [len(urlparse(u).netloc.split('.')) for u in urls]
            url_feats = [np.mean(lens), np.mean(dots), has_at, has_ip, np.mean(subdomains), len(urls)]

        # 构造特征字典
        feats_dict = {
            'body': body,
            'sentiment': sentiment,
            'subjectivity': subjectivity,
            'avg_url_len': url_feats[0],
            'avg_url_dots': url_feats[1],
            'has_at_symbol': url_feats[2],
            'has_ip_url': url_feats[3],
            'avg_subdomains': url_feats[4],
            'url_count': url_feats[5]
        }
        return feats_dict, urls


# 初始化推理引擎
engine = InferenceEngine()

# 4. Gradio
def predict_interface(file, engine_type):
    if file is None:
        return "未上传文件", "0%", "无"

    try:
        data, urls = engine.extract_eml_data(file.name)

        if engine_type == "机器学习引擎 (Random Forest)":
            input_df = pd.DataFrame([data])
            prob = engine.rf_pipeline.predict_proba(input_df)[0][1]

        else:
            inputs = engine.tokenizer.encode_plus(
                data['body'], max_length=512, padding='max_length',
                truncation=True, return_tensors='pt'
            ).to(DEVICE)

            num_tensor = torch.tensor([
                [data[c] for c in NUMERIC_COLS]
            ], dtype=torch.float).to(DEVICE)

            with torch.no_grad():
                outputs = engine.bert_model(inputs['input_ids'], inputs['attention_mask'], num_tensor)
                prob = torch.softmax(outputs, dim=1)[0][1].item()

        # 结果判定
        if prob < 0.5:
            res, level = "✅ 安全：正常邮件", "无风险"
        elif 0.5 <= prob < 0.75:
            res, level = "🟠 中危：风险钓鱼邮件", "中危"
        else:
            res, level = "🛑 高危：极高风险钓鱼邮件", "高危"

        return f"{res} ({level})", f"{prob:.2%}", "\n".join(urls) if urls else "无链接"

    except Exception as e:
        return f"处理失败: {str(e)}", "Error", "Error"


# --- 5. 构建 UI ---
with gr.Blocks(title="AI 钓鱼邮件多引擎检测", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ 钓鱼邮件 AI 多引擎检测系统")
    gr.Markdown("结合传统特征工程 (Random Forest) 与深度学习语义分析 (DistilBERT) 的混合检测方案。")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="上传 .eml 邮件", file_types=[".eml"])
            engine_selector = gr.Radio(
                choices=["机器学习引擎 (Random Forest)", "深度学习引擎 (DistilBERT)"],
                value="机器学习引擎 (Random Forest)",
                label="选择检测引擎"
            )
            analyze_btn = gr.Button("开始 AI 分析", variant="primary")

        with gr.Column(scale=1):
            res_output = gr.Textbox(label="检测结论", interactive=False)
            prob_output = gr.Textbox(label="恶意概率评分", interactive=False)
            url_output = gr.Textbox(label="提取到的链接列表", lines=8, interactive=False)

    analyze_btn.click(
        fn=predict_interface,
        inputs=[file_input, engine_selector],
        outputs=[res_output, prob_output, url_output]
    )

if __name__ == "__main__":
    demo.launch()
