import gradio as gr
import os
from predict import EMLPredictor
import time
import signal

# 1. 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RF_MODEL_PATH = os.path.join(BASE_DIR, 'phishing_detector_final.pkl')
BERT_MODEL_PATH = os.path.join(BASE_DIR, 'phishing_bert_model.pth')

# 2. 初始化预测器
predictor = EMLPredictor(RF_MODEL_PATH, BERT_MODEL_PATH)

def process_email(file, engine_mode):
    if file is None:
        return "请先上传邮件文件", "0%", "无"

    try:
        # 映射 UI 选项到代码内部模式
        mode_map = {
            "机器学习引擎": "RF",
            "深度学习引擎": "BERT",
            "加权集成检测": "ENSEMBLE"
        }
        selected_mode = mode_map[engine_mode]

        # 调用 predict 方法加权
        prob = predictor.predict(file.name, mode=selected_mode, rf_weight=0.4, bert_weight=0.6)

        # 提取 URL
        _, urls = predictor._extract_eml_content(file.name)

        # 结果分级判定
        if prob < 0.5:
            res, level = "✅ 安全：正常邮件", "无风险"
        elif 0.5 <= prob < 0.75:
            res, level = "🟠 中危：风险钓鱼邮件", "中危"
        else:
            res, level = "🛑 高危：极高风险钓鱼邮件", "高危"

        return f"{res} ({level})", f"{prob:.2%}", "\n".join(urls) if urls else "未检测到链接"

    except Exception as e:
        return f"检测出错: {str(e)}", "Error", "Error"


def exit_system():
    time.sleep(0.8)
    os.kill(os.getpid(), signal.SIGINT)
    return None

# 3. 构建 UI
with gr.Blocks(title="AI 钓鱼邮件多引擎检测", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ 钓鱼邮件 AI 多引擎检测系统")
    gr.Markdown("结合传统特征工程 (RF) 与深度学习语义分析 (BERT) 的混合检测方案。")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="上传 .eml 邮件内容", file_types=[".eml"])

            # 功能选择
            engine_selector = gr.Radio(
                choices=["机器学习引擎", "深度学习引擎", "加权集成检测"],
                value="加权集成检测",
                label="选择检测引擎"
            )

            analyze_btn = gr.Button("开始 AI 智能分析", variant="primary")
            exit_btn = gr.Button("退出系统", variant="stop", min_width=100)

        with gr.Column(scale=1):
            res_output = gr.Textbox(label="检测结论及风险等级", interactive=False)
            prob_output = gr.Textbox(label="恶意概率评分", interactive=False)
            url_output = gr.Textbox(label="提取到的可疑链接列表", lines=8, interactive=False)

    analyze_btn.click(
        fn=process_email,
        inputs=[file_input, engine_selector],
        outputs=[res_output, prob_output, url_output]
    )
    exit_btn.click(fn=exit_system,
                   js="window.close()")

if __name__ == "__main__":
    demo.launch()
