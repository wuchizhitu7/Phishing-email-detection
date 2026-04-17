# A dual-engine detection system for phishing emails based on multi-dimensional feature engineering
### 基于多维度特征工程的钓鱼邮件双引擎检测系统

这是一个基于机器学习与深度学习混合架构的自动化邮件安全检测平台。系统结合了传统的**随机森林 (Random Forest)** 特征工程方法与现代的**DistilBERT**语义分析技术，为用户提供多维度的风险评估。

---

# 🛡️ 钓鱼邮件 AI 多引擎检测系统

## 🌟 核心特性

* **双引擎架构**：支持在传统的随机森林流水线（TF-IDF + 专家特征）与先进的 BERT 融合模型之间一键切换。
* **集成检测**：平衡RF对URL结构特征效果好而对语义不敏感的问题。
* **多维特征工程**：
    * **语义分析**：通过 TextBlob 提取邮件的情感极性与主观性。
    * **URL 审计**：自动扫描正文链接，计算平均长度、点号频率、子域名深度及 IP 格式检测。
* **自动化流水线**：从 `.mbox` 原始数据清洗、HTML 去标签、特征提取到模型训练的全自动流程。
* **交互式 UI**：基于 Gradio 构建，支持实时检测结论、恶意概率评分、可疑链接提取及系统一键关闭功能。

---

## 🏗️ 系统架构


1.  **数据层** (`data_clean.py`)：处理邮件编码、递归解析多部分邮件，执行深度文本清洗。
2.  **特征层** (`features_extractor.py`)：融合 NLP 特征与统计特征。
3.  **模型层** (`RF.py`, `BERT.py`)：
    * **RF 引擎**：结合 TF-IDF 关键词向量与数值特征。
    * **BERT 引擎**：利用 DistilBERT 提取 768 维语义向量，并与数值特征在隐藏层进行拼接（Concatenation）。
4.  **应用层** (`app_gradio.py`)：通过 `EMLPredictor` 类实现跨模型的高效推理。

---

## 📂 项目结构

| 文件 | 说明 |
| :--- | :--- |
| **`data_clean.py`** | 原始 `.mbox` 数据解析、HTML 剥离、控制字符过滤。 |
| **`features_extractor.py`** | 提取专家特征（URL 统计、情感分析），生成增强型数据集。 |
| **`RF.py`** | 随机森林流水线训练脚本，包含 TF-IDF 预处理器。 |
| **`BERT.py`** | DistilBERT 混合模型定义（BERT + MLP）及训练逻辑。 |
| **`predict.py`** | 统一预测封装类，负责加载模型、解析 `.eml` 并执行推理。 |
| **`app_gradio.py`** | 可视化界面：包含双引擎切换、集成检测、分析展示。 |
| **`phishing_detector_final.pkl`** | 存储随机森林分类器判定钓鱼邮件的所有逻辑规则与文本特征权重。 |
| **`numeric_features_list.pkl`** | 记录模型输入特征的精确顺序，确保预测时提取的数值数据能被模型正确识别。 |
| **`phishing_bert_model.pth`** | 训练好的 DistilBERT 权重，存储了模型对文本语义理解的深层规律。 |

---

## 🚀 快速开始

### 1. 环境安装
建议使用 Python 3.8+ 环境：
```bash
pip install torch transformers scikit-learn pandas textblob beautifulsoup4 gradio joblib
```

### 2. 数据准备与模型训练
若需重新训练模型，请按顺序执行：
1.  **特征提取**：生成训练所需的 `.csv` 文件。
    ```bash
    python features_extractor.py
    ```
2.  **训练 RF 引擎**：
    ```bash
    python RF.py
    ```
3.  **训练 BERT 引擎**：
    ```bash
    python BERT.py
    ```

### 3. 运行检测界面
```bash
python app_gradio.py
```

---

## 🔗 数据集说明

本项目通过组合两个著名的开源数据集来平衡样本分布：

- **正常邮件 (Ham)**: 选自 **Enron Email Dataset**。代表真实的企业商务通讯，包含大量非恶意但具有复杂结构的专业邮件。
- **钓鱼邮件 (Phishing)**: 选自 **Nazario Phishing Dataset**。包含多年来收集的各种真实钓鱼样本（如 PayPal、银行、社交账户伪冒）。

---

## 📊 模型性能评估

本系统在独立测试集（总数据 20%）上进行了严格评估。以下是 随机森林 (Random Forest) 与 DistilBERT 引擎的性能对比。

| 评估指标 | RF | DistilBERT |
| :--- | :--- | :--- |
| **`准确率`** | 0.99 | 0.97 |
| **`F1分数`** | 0.98 | 0.94 |
| **`测试样本数量`** | 1168 | 1168 |

---

## 🛠️ 技术细节

### 特征融合 (Feature Fusion)
在 BERT 模型中，我们采用了 **混合输入架构**：
* **文本流**：$`Text \xrightarrow{BERT} Vector_{768}`$
* **数值流**：$`Stats_{8} \xrightarrow{Scaling} Features_{8}`$
* **融合层**：$`Concatenate(Vector_{768}, Features_{8}) \xrightarrow{MLP} Output_{2}`$

### 风险评估等级
系统根据预测概率 $`P`$ 进行智能判定：
* **安全 (Normal)**：$`P < 0.5`$
* **中危 (Warning)**：$`0.5 \le P < 0.75`$
* **高危 (High Risk)**：$`P \ge 0.75`$

---

## 📸 系统运行截图

<img width="1803" height="642" alt="界面" src="https://github.com/user-attachments/assets/bce5e9dc-5582-48bf-8add-95b9260eb33c" />
图 1：基于 Gradio 搭建的交互式上传与分析界面

<img width="1761" height="806" alt="RF" src="https://github.com/user-attachments/assets/b9f92295-fc3f-434a-bec1-d67eea3ff8cb" />
图 2：RF引擎检测

<img width="1722" height="808" alt="BERT" src="https://github.com/user-attachments/assets/2335fe1b-a482-440f-91fb-54d8c0401157" />
图 3：BERT引擎检测

<img width="1736" height="804" alt="集成" src="https://github.com/user-attachments/assets/7fefa6cd-a186-4d8e-9d30-31866e493ea0" />
图 4：集成检测

---



