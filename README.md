# Multi-Dimensional Phishing Email Detection System
### 基于多维度特征工程的钓鱼邮件深度检测系统

本项目提供了一个从原始邮件数据解析到实时恶意判定的全流程机器学习解决方案。通过对邮件正文进行 **NLP 语义分析** 以及对链接进行 **URL 结构词法分析**，系统能够有效识别各类复杂的钓鱼攻击。

---

## 🌟 项目亮点

- **深度特征挖掘**：不仅依赖简单的关键词匹配，还深入分析了 URL 的子域名深度、IP 隐写、点号密度等 8 维词法结构特征。
- **消除特征泄露**：针对 Enron 与 Nazario 数据集存在的“特征泄露”（如 `received_count`）进行了主动剔除，确保模型学习的是钓鱼行为的本质逻辑，而非数据集本身的标签差异。
- **多模态融合**：通过 `ColumnTransformer` 将 TF-IDF 文本向量化特征与手工构造的数值特征（语义情感、URL 复杂度）完美融合。
- **全流程实战**：涵盖从原始 mbox 解析、数据清洗、特征提取、模型训练、5 折交叉验证到最终 `.eml` 文件推理的完整链路。

---

## 📂 文件说明

| 文件名 | 功能描述 |
| :--- | :--- |
| `data_clean.py` | **数据清洗层**：解析 mbox 原始文件，处理 Unicode 及 HTML 乱码，提取 Header、Body 及原始 URL。 |
| `features_extractor.py` | **特征工程层**：利用 `TextBlob` 进行情感分析，并对 URL 的词法结构进行多维度量化提取。 |
| `model.py` | **模型训练层**：构建混合特征处理流水线，执行交叉验证，生成最终的随机森林 `.pkl` 模型。 |
| `predict.py` | **推理应用层**：支持解析标准 `.eml` 邮件文件，自动完成特征对齐并给出实时恶意概率评分。 |
| `enriched_emails_dataset.csv` | 包含所有手工提取特征及标签的结构化数据集，可直接用于模型复现。 |
| `phishing_detector_final.pkl` | 经过训练且包含预处理逻辑（Pipeline）的持久化分类模型文件。 |

---

## 🛠️ 技术栈

- **数据处理**: `Pandas`, `NumPy`, `mailbox`
- **自然语言处理**: `TextBlob` (语义分析), `BeautifulSoup4` (HTML 提取), `Scikit-learn (TfidfVectorizer)`
- **机器学习**: `Scikit-learn (RandomForestClassifier, Pipeline, ColumnTransformer)`
- **模型导出**: `Joblib`

---

## 📊 实验表现

在主动剔除高权重“泄露”特征并进行 5 折交叉验证后，模型表现极其稳健，证明了特征工程的有效性：

- **平均准确率 (Accuracy)**: 99.06% (± 0.0022)
- **F1-Score (恶意邮件)**: 0.99
- **核心贡献特征 (Importance TOP 5)**:
  1. `avg_url_len` (URL 平均长度)
  2. `avg_subdomains` (子域名复杂度)
  3. `avg_url_dots` (URL 点号密度)
  4. `url_count` (链接总数)
  5. `enron` (TF-IDF 识别出的商务场景词汇)

---

## 🚀 快速开始

### 1. 环境准备

确保您的 Python 环境已安装以下依赖库：

```bash
pip install pandas numpy scikit-learn textblob beautifulsoup4 joblib
```

### 2. 数据处理与特征提取

若要从原始 mbox 文件重新生成数据集，请依次运行：

```bash
python data_clean.py          # 步骤1：清洗原始邮件并去重
python features_extractor.py   # 步骤2：执行深度语义与 URL 特征提取
```

### 3. 训练与验证

运行训练脚本以执行交叉验证、生成分类报告并保存模型：

```bash
python model.py
```

### 4. 实时预测 EML 邮件

您可以直接对任何导出的 `.eml` 邮件进行检测：

```bash
python predict.py
# 运行后按提示输入 EML 文件路径（如：test_sample.eml）
```

---

## 🔗 数据集说明

本项目通过组合两个著名的开源数据集来平衡样本分布：

- **正常邮件 (Ham)**: 选自 **Enron Email Dataset**。代表真实的企业商务通讯，包含大量非恶意但具有复杂结构的专业邮件。
- **钓鱼邮件 (Phishing)**: 选自 **Nazario Phishing Dataset**。包含多年来收集的各种真实钓鱼样本（如 PayPal、银行、社交账户伪冒）。

---

**Author**: Your GitHub ID  
**Project Date**: 2026
```
