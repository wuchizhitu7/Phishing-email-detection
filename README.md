# 🛡️ 钓鱼邮件智能检测系统

基于 **Random Forest + BERT 双引擎混合架构** 的中文钓鱼邮件检测系统，支持传统特征工程与深度学习语义分析的加权融合推理，具备 AI 生成钓鱼邮件检测能力。

## ✨ 特性

- **双引擎混合检测**：RF 特征工程 + BERT 语义理解，加权融合决策
- **AI 生成钓鱼识别**：训练数据含生成式 AI 钓鱼邮件，支持对抗检测
- **Perplexity 困惑度特征**：统计层辅助判断文本是否由 AI 生成
- **多模式推理**：支持 RF 单独、BERT 单独、ENSEMBLE 加权三种模式
- **完整工程链路**：数据清洗 → 特征提取 → 训练 → 评估 → Web UI
- **多语种架构**：核心代码中/英文双轨，模型可互换

## 📁 项目结构

```
Email Check/
├── data_clean.py            # 数据清洗：mbox、TREC06C 等多格式解析
├── features_extractor.py    # 特征提取：语义 + URL + perplexity
├── RF.py                    # Random Forest 训练（jieba 分词 + TF-IDF）
├── BERT.py                  # DistilBERT / bert-base-chinese 混合模型训练
├── predict.py               # 推理引擎：RF + BERT + 加权集成
├── evaluate_model.py        # 5 折交叉验证评估
├── tokenizer_utils.py       # jieba 分词器共享模块
├── generate_ai_phishing.py  # AI 钓鱼邮件批量生成工具
├── app_gradio.py            # Gradio Web 界面
├── test.py                  # EML 文件查看工具
├── bert-base-chinese/       # 中文 BERT 预训练模型
├── phishing_detector_final.pkl  # 训练好的 RF 模型
├── phishing_bert_model.pth      # 训练好的 BERT 权重
└── README.md
```

## 🚀 快速开始

### 环境要求

```
Python >= 3.10
CUDA >= 11.8（GPU 推理推荐）
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

支持上传 `.eml` 文件，选择检测引擎，实时输出判定结论。

### 命令行检测

编辑 `predict.py` 底部 `target_eml` 路径，运行：

```bash
python predict.py
```

## 🧠 整体框架

```
┌──────────────────────────────────────────────────────────────┐
│                        数据层 Data Layer                       │
├──────────────┬──────────────┬──────────────┬─────────────────┤
│   TREC06C    │  CHIFRAUD    │  AI-Generated │   用户 EML      │
│  (64K 中文)  │ (384K 欺诈)  │   (5K 钓鱼)   │   (单文件推理)   │
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
              │  · 关键词    │  中文钓鱼关键词密度     │
              └──────┬──────┘                         │
                     │                                │
       ┌─────────────┼─────────────┐                  │
       │             │             │                  │
┌──────▼──────┐ ┌────▼────┐ ┌─────▼──────┐           │
│  RF 模型    │ │  BERT   │ │  推理引擎   │◄──────────┘
│  · TF-IDF   │ │  · 解冻 │ │  · RF       │  predict.py
│  · jieba    │ │  顶层4层 │ │  · BERT     │
│  · 100棵树  │ │  · AMP  │ │  · ENSEMBLE │
│             │ │  · 差分 │ │  · 加权融合  │
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
| mbox 解析 | 支持 Enron / 钓鱼邮件 mbox 格式 |
| TREC06C 解析 | 支持 TREC 中文垃圾邮件语料库 index + 单文件格式 |
| 编码清洗 | BeautifulSoup HTML 剥离 + Unicode 控制字符过滤 |
| URL 提取 | 正则匹配 `http/https` 链接 |

#### 2. 特征提取 (`features_extractor.py`)

| 特征类别 | 特征列表 | 工具 |
|------|------|------|
| **语义特征** | `sentiment`（情感极性）、`subjectivity`（关键词密度） | SnowNLP / jieba |
| **URL 结构** | `avg_url_len`、`avg_url_dots`、`has_at_symbol`、`has_ip_url`、`avg_subdomains`、`url_count` | urlparse |
| **统计特征** | `perplexity`（困惑度，AI 文本偏低） | BERT MLM |
| **中文关键词** | 30 个钓鱼高频词（验证/账号/银行/冻结/中奖/补贴...） | 自定义词典 |

#### 3. 双引擎模型

| 引擎 | 模型 | 特点 |
|------|------|------|
| **RF** | Random Forest（100 棵） + TF-IDF（500 维）+ jieba 分词 | 特征透明、训练快、URL 结构敏感 |
| **BERT** | bert-base-chinese（12 层，解冻顶层 4 层）+ MLP 融合层 | 语义理解、上下文感知、差分学习率 |
| **ENSEMBLE** | RF（40%）+ BERT（60%）加权平均 | 取长补短、工业级鲁棒性 |

#### 4. BERT 训练优化

| 策略 | 配置 |
|------|------|
| 分层解冻 | 冻结底层 8 层 → 解冻顶层 4 层 |
| 差分学习率 | BERT 骨干 2e-5 / 分类头 1e-4 |
| 混合精度 | AMP (FP16) 训练加速 ~1.5x |
| 序列截断 | MAX_LEN=256（中文邮件覆盖率高） |
| 防过拟合 | Dropout 0.3 |

#### 5. 推理引擎 (`predict.py`)

- 解析 `.eml` 文件 → 提取正文 + URL + 特征
- 三种模式：`RF` / `BERT` / `ENSEMBLE`
- 输出：恶意概率 + 风险等级 + 可疑链接列表

#### 6. Web 界面 (`app_gradio.py`)

- Gradio 可视化界面
- 上传 EML → 选择引擎 → 一键分析
- 实时展示检测结论、概率评分、链接列表
- 点击「退出系统」→ 关闭服务并退出进程
- 关闭浏览器 → 终端回车停止后端

## 📊 评估指标

采用 **5 折分层交叉验证**，每折独立训练 RF / 加载已训练 BERT 权重评测：

| 指标 | Random Forest | DistilBERT |
|------|:---:|:---:|
| **Accuracy** | 0.9825 ± 0.0009 | **0.9962 ± 0.0002** |
| **F1-Score** | 0.9772 ± 0.0012 | **0.9951 ± 0.0003** |
| 稳定性 (σ) | ±0.0012 | **±0.0003** |
| 训练时间 | ~3 分钟 | ~2 小时 (GPU) |

> **数据规模**：16.2 万条中文邮件（正常 99,513 / 欺诈 62,863），测试集 32,476 条

## 📦 数据来源

| 数据集 | 年份 | 语言 | 规模 | 用途 |
|------|:---:|:---:|:---:|------|
| [TREC06C](https://plg.uwaterloo.ca/cgi-bin/cgiwrap/gvcormac/foo06) | 2006 | 中文 | 19,500 条 | 正常邮件（ham） |
| [CHIFRAUD](https://github.com/xuemingxxx/ChiFraud) | 2025 | 中文 | ~384,000 条 | 当代欺诈文本（11 类欺诈） |
| AI 生成钓鱼 | 2025 | 中文 | 4,969 条 | 覆盖 5 种钓鱼类型 × 多种语气 |

**数据处理策略**：
- CHIFRAUD 正常类采样至 80,000 条，防止类别失衡
- TREC06C 仅保留 ham（正常），丢弃 2006 年旧 spam
- AI 钓鱼覆盖账号安全、福利补贴、快递物流、企业冒名、退款理赔 5 类

## 📸 运行截图

> 将截图保存在 `screenshots/` 目录下，按以下链接引用。

### Web 界面

| 场景 | 截图 | 说明 |
|------|------|------|
| 主页 | `screenshots/web_main.png` | 上传 EML + 选择引擎 + 分析按钮 |
| 安全邮件 | `screenshots/web_safe.png` | 检测结果为安全、低风险 |
| 高风险邮件 | `screenshots/web_danger.png` | 检测结果为高危钓鱼 |
| ENSEMBLE 模式 | `screenshots/web_ensemble.png` | 加权融合下的多引擎报告 |

### 命令行推理

| 场景 | 截图 | 说明 |
|------|------|------|
| RF 引擎 | `screenshots/cli_rf.png` | RF 单独推理输出 |
| BERT 引擎 | `screenshots/cli_bert.png` | BERT 单独推理输出 |
| ENSEMBLE | `screenshots/cli_ensemble.png` | 三模式并行对比 |

### 评估报告

| 场景 | 截图 | 说明 |
|------|------|------|
| 5 折 CV | `screenshots/eval_cv.png` | 交叉验证完整输出 |
| 特征重要性 | `screenshots/eval_features.png` | RF 贡献度前 10 特征 |

## 🔧 命令行参考

```bash
# 特征提取（中文模式）
python features_extractor.py cn

# 训练 RF
python RF.py cn

# 训练 BERT
python BERT.py

# 交叉验证评估
python evaluate_model.py

# 查看 EML 文件内容
python test.py predict_email.eml

# 生成 AI 钓鱼训练数据（需 API Key）
python generate_ai_phishing.py
```

## 📝 开发历程

| 阶段 | 工作 | 数据规模 | BERT F1 |
|:---:|------|:---:|:---:|
| 一 | 英文双引擎搭建（Enron + SA + Nazario） | 22K | 0.9953 |
| 二 | 中文本地化（bert-base-chinese + TREC06C + jieba） | 29K | 0.9588 |
| 三 | 引入 CHIFRAUD 当代欺诈数据 | 112K | — |
| 四 | 加入 AI 生成钓鱼 + Perplexity 特征 | 162K | 0.9951 |

## 📄 License

MIT License
