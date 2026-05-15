# 🛡️ 中文钓鱼邮件智能检测系统

> 基于 **Random Forest + BERT 双引擎混合架构**，融入 **Perplexity 困惑度特征**，具备 AI 生成钓鱼邮件检测能力。

## ✨ 特性

- **双引擎混合检测**：RF（jieba 分词 + TF-IDF）+ BERT（bert-base-chinese 语义理解），ENSEMBLE 加权融合
- **AI 生成钓鱼识别**：训练集含 5000 封 LLM 生成的钓鱼邮件，支持对抗检测
- **Perplexity 困惑度**：BERT MLM 计算文本困惑度作为统计特征，辅助识别 AI 生成文本
- **多模式推理**：RF 单独 / BERT 单独 / ENSEMBLE 加权
- **5 折交叉验证**：真实泛化评估，非单次 split
- **Gradio Web 界面**：上传 EML → 一键分析，关闭浏览器自动停止后端

## 📁 项目结构

```
Email Check/
├── data_clean.py              # 数据清洗：TREC06C 格式解析
├── features_extractor.py      # 特征提取：语义 + URL + perplexity
├── RF.py                      # Random Forest 训练（jieba + TF-IDF + 9 维特征）
├── BERT.py                    # bert-base-chinese 混合模型训练（AMP + 差分学习率）
├── predict.py                 # 推理引擎：RF + BERT + ENSEMBLE
├── evaluate_model.py          # 5 折交叉验证评估
├── ablation_experiments.py    # 消融实验（Perplexity 特征对比）
├── tokenizer_utils.py         # jieba 分词器共享模块
├── generate_ai_phishing.py    # AI 钓鱼邮件训练数据批量生成
├── generate_ai_test.py        # AI 钓鱼邮件测试数据生成（消融专用）
├── integrate_chifraud.py      # CHIFRAUD 数据处理
├── app_gradio.py              # Gradio Web 界面
└── README.md
```

> 模型权重（`.pth`、`.pkl`）、数据集（`data_trec06c/`、`chifraud_data/`）、预训练模型（`bert-base-chinese/`）因大小限制未纳入版本控制。

## 🚀 快速开始

### 环境要求

```
Python >= 3.10
CUDA >= 11.8（GPU 推理推荐，4GB 显存可跑）
```

### 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pandas scikit-learn jieba snownlp tqdm gradio beautifulsoup4 joblib
```

### 下载模型

```bash
python -c "
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('bert-base-chinese')
tok.save_pretrained('bert-base-chinese')
model.save_pretrained('bert-base-chinese')
"
```

### 启动 Web 界面

```bash
python app_gradio.py
```

### 命令行检测

编辑 `predict.py` 底部 `target_eml` 路径，运行：

```bash
python predict.py
```

## 🧠 整体框架

```
┌──────────────────────────────────────────────────────────────┐
│                        数据层 Data Layer                      │
├──────────────┬──────────────┬──────────────┬─────────────────┤
│   TREC06C    │  CHIFRAUD    │  AI-Generated │   用户 EML      │
│  (19K 中文)  │ (60K 采样)   │  (5K 钓鱼)    │   (单文件推理)   │
└──────┬───────┴──────┬───────┴──────┬───────┴────────┬────────┘
       │              │              │                │
       └──────────────┴──────────────┘                │
                     │                                │
              ┌──────▼──────┐                         │
              │  数据清洗    │  data_clean.py          │
              │  · 编码修复  │                         │
              │  · HTML剥离  │                         │
              │  · URL 提取  │                         │
              │  · 去重处理  │                         │
              └──────┬──────┘                         │
                     │                                │
              ┌──────▼──────┐                         │
              │  特征工程    │  features_extractor.py  │
              │  · 语义特征  │  SnowNLP 情感           │
              │  · URL 特征  │  urlparse 结构          │
              │  · 困惑度    │  BERT MLM perplexity   │
              │  · 关键词    │  中文钓鱼关键词(30个)   │
              └──────┬──────┘                         │
                     │                                │
       ┌─────────────┼─────────────┐                  │
       │             │             │                  │
┌──────▼──────┐ ┌────▼────┐ ┌─────▼──────┐           │
│  RF 模型    │ │  BERT   │ │  推理引擎   │◄──────────┘
│  · TF-IDF   │ │  · 解冻 │ │  · RF       │  predict.py
│  · jieba    │ │  顶层4层 │ │  · BERT     │
│  · 100棵树  │ │  · AMP  │ │  · ENSEMBLE │
│  · 9维特征  │ │  · 差分 │ │  · 加权融合  │
│             │ │  学习率  │ │             │
└──────┬──────┘ └────┬────┘ └──────┬──────┘
       │             │             │
       └─────────────┼─────────────┘
                     │
              ┌──────▼──────┐
              │   输出层     │
              │ · 恶意概率   │
              │ · 风险等级   │
              │ · 可疑链接   │
              │ · 引擎报告   │
              └─────────────┘
```

### 核心模块详解

#### 1. 数据清洗 (`data_clean.py`)

| 功能 | 说明 |
|------|------|
| TREC06C 解析 | 支持 TREC 中文垃圾邮件语料库（index + 单文件格式，64,620 条标注） |
| 编码清洗 | BeautifulSoup HTML 剥离 + Unicode 控制字符过滤 + 空白规范化 |
| URL 提取 | 正则匹配 `http/https` 链接 |

#### 2. 特征提取 (`features_extractor.py`)

提取 **9 维数值特征**（8 维语义/URL + 1 维统计），与 TF-IDF 文本特征拼接：

| 特征类别 | 特征 | 维度 | 工具 |
|------|------|:---:|------|
| 语义 | `sentiment`（情感极性）、`subjectivity`（关键词密度） | 2 | SnowNLP |
| URL 结构 | `avg_url_len`、`avg_url_dots`、`has_at_symbol`、`has_ip_url`、`avg_subdomains`、`url_count` | 6 | urlparse |
| 统计 | `perplexity`（困惑度，log 归一化，AI 文本偏低） | 1 | BERT MLM |

#### 3. 双引擎模型

| 引擎 | 架构 | 关键配置 |
|------|------|------|
| **RF** | TF-IDF(500维) + 9维数值 → RandomForest(100棵) | jieba 分词、class_weight='balanced' |
| **BERT** | bert-base-chinese(12层) → [CLS] + 9维数值 → FC(256) → FC(2) | 冻结底层8层、解冻顶层4层 |
| **ENSEMBLE** | RF(40%) + BERT(60%) 加权平均 | 工业级鲁棒性 |

#### 4. BERT 训练优化

| 策略 | 配置 | 目的 |
|------|------|------|
| 分层解冻 | 冻结底层 8 层 → 解冻顶层 4 层 | BERT 学习钓鱼语义，保留底层语法 |
| 差分学习率 | backbone 2e-5 / head 1e-4 | 防灾难性遗忘 + 加速分类头收敛 |
| 混合精度 | AMP (FP16) | 训练加速 ~1.5x，降低显存 |
| 序列截断 | MAX_LEN=256 | 适配 4GB 显存 |
| 防过拟合 | Dropout 0.3 | — |
| 困惑度归一化 | `log1p(perplexity)` | 将 27K~136K 压缩到 ~10~12，避免 FP16 NaN |

## 📊 评估指标

5 折分层交叉验证 — 数据量 84,482 条（正常 49,513 / 欺诈 34,969）：

| 指标 | Random Forest | bert-base-chinese |
|------|:---:|:---:|
| **Accuracy** | 0.9782 ± 0.0005 | **0.9967 ± 0.0005** |
| **F1-Score** | 0.9735 ± 0.0006 | **0.9960 ± 0.0006** |
| 稳定性 (σ) | ±0.0006 | **±0.0006** |
| 训练时间 | ~3 分钟 | ~2 小时 (GPU) |

### 消融实验：Perplexity 特征

| 实验 | BERT F1 | 说明 |
|------|:---:|------|
| 含 perplexity | 0.9975 | 混合测试集（200 AI + 200 正常） |
| 不含 perplexity | 待跑 | 需生成无 perplexity 数据集 + 重训 |

## 📦 数据来源

| 数据集 | 年份 | 规模（使用） | 用途 |
|------|:---:|:---:|------|
| [TREC06C](https://plg.uwaterloo.ca/cgi-bin/cgiwrap/gvcormac/foo06) | 2006 | 19,500 条 | 正常中文邮件（仅保留 ham） |
| [CHIFRAUD](https://github.com/xuemingxxx/ChiFraud) | 2025 | 各 30,000 条 | 正常 + 欺诈文本采样 |
| AI 生成钓鱼 | 2025 | 4,969 条 | 5 种钓鱼类型 × 多种语气 |

## 🔧 命令行参考

```bash
# 特征提取
python features_extractor.py

# 训练 RF
python RF.py

# 训练 BERT
python BERT.py

# 5 折交叉验证
python evaluate_model.py

# 消融实验（需先生成 AI 测试数据）
python generate_ai_test.py
python ablation_experiments.py

# 查看 EML 内容
python test.py predict_email.eml

# 生成 AI 钓鱼训练数据（需 API Key）
python generate_ai_phishing.py
```

## 📄 License

MIT License
