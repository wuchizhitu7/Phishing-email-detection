import gradio as gr
import joblib
import time
import os
import re
import webbrowser
import numpy as np
import pandas as pd
from gradio import close_all
from textblob import TextBlob
from urllib.parse import urlparse
from predict import EMLPredictor

# --- 路径处理 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'phishing_detector_final.pkl')

# --- 初始化预测器 ---
try:
    predictor = EMLPredictor(MODEL_PATH)
except Exception as e:
    print(f"错误：无法加载模型文件: {e}")


def process_email(file):
    if file is None:
        return "未上传文件", "0%", "无"

    try:
        # 1. 提取内容
        body, urls = predictor._extract_eml_content(file.name)

        # 2. 特征计算
        blob = TextBlob(body)
        sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if not urls:
            url_feats = [0, 0, 0, 0, 0, 0]
        else:
            lens = [len(u) for u in urls]
            dots = [u.count('.') for u in urls]
            has_at = 1 if any('@' in u for u in urls) else 0
            has_ip = 1 if any(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}', u) for u in urls) else 0
            subdomains = [len(urlparse(u).netloc.split('.')) for u in urls]
            url_feats = [np.mean(lens), np.mean(dots), has_at, has_ip, np.mean(subdomains), len(urls)]

        # 3. 构造推理数据
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
        prob = predictor.model.predict_proba(input_df)[0][1]

        # --- 细分类逻辑 ---
        if prob < 0.5:
            res = "✅ 安全：正常邮件"
            risk_level = "无风险"
        elif 0.5 <= prob < 0.6:
            res = "🔵 低危：疑似钓鱼邮件"
            risk_level = "低危"
        elif 0.6 <= prob < 0.7:
            res = "🟠 中危：风险钓鱼邮件"
            risk_level = "中危"
        else:
            res = "🛑 高危：极高风险钓鱼邮件"
            risk_level = "高危"

        return f"{res} ({risk_level})", f"{prob:.2%}", "\n".join(urls) if urls else "无链接"

    except Exception as e:
        return f"处理失败: {str(e)}", "Error", "Error"


# --- 退出功能函数 ---
def exit_app():
    print("正在关闭系统...")
    time.sleep(0.5)
    os._exit(0)


# --- 构建界面 ---
with gr.Blocks(title="邮件安全检测") as demo:
    with gr.Row():
        gr.Markdown("# 🛡️ 钓鱼邮件 AI 检测系统")
        # 添加退出按钮，使用 stop 变体颜色
        exit_btn = gr.Button("关闭系统", variant="stop", scale=0)

    with gr.Row():
        file_input = gr.File(label="上传 .eml 邮件", file_types=[".eml"])

    with gr.Row():
        res_output = gr.Textbox(label="检测结论及风险等级", interactive=False)
        prob_output = gr.Textbox(label="恶意概率", interactive=False)

    url_output = gr.Textbox(label="链接提取列表", lines=5)

    analyze_btn = gr.Button("开始 AI 分析", variant="primary")

    close_script = """
        () => {
            window.opener = null;
            window.open('', '_self');
            window.close();
            setTimeout(() => {
                window.location.href = "about:blank";
            }, 100);
        }
        """

    # 绑定功能
    analyze_btn.click(fn=process_email, inputs=file_input, outputs=[res_output, prob_output, url_output])
    exit_btn.click(fn=exit_app,inputs=None, outputs=None,js=close_script)

if __name__ == "__main__":
    print("系统已启动，正在打开浏览器...")
    demo.launch(inbrowser=True)