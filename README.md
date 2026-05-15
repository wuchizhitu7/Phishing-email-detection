# 🛡️ 中文钓鱼邮件智能检测系统

> 基于 **Random Forest + BERT 双引擎混合架构**，融入 **Perplexity 困惑度特征**，具备 AI 生成钓鱼邮件检测能力。

## ✨ 特性

- **双引擎混合检测**：RF（jieba 分词 + TF-IDF）+ BERT（bert-base-chinese 语义理解），ENSEMBLE 加权融合
- **AI 生成钓鱼识别**：训练集含 5000 封 LLM 生成的钓鱼邮件，支持对抗检测
- **Perplexity 困惑度**：BERT MLM 计算文本困惑度作为统计特征，辅助识别 AI 生成文本
- **多模式推理**：RF 单独 / BERT 单独 / ENSEMBLE 加权
- **5 折交叉验证**：真实泛化评估，非单次 split
- **Gradio Web 界面**：上传 EML → 一键分析，关闭浏览器自动停止后端
- **URL 安全检测**：本地规则评级链接风险，后处理加权修正模型概率

## 📁 项目结构

```
Email Check/
├── data_clean.py              # 数据清洗：TREC06C 格式解析
├── features_extractor.py      # 特征提取：语义 + URL + perplexity
├── RF.py                      # Random Forest 训练（jieba + TF-IDF + 9 维特征）
├── BERT.py                    # bert-base-chinese 混合模型训练（AMP + 差分学习率）
├── predict.py                 # 推理引擎：RF + BERT + ENSEMBLE + URL后处理
├── evaluate_model.py          # 5 折交叉验证评估
├── ablation_experiments.py    # 消融实验（Perplexity 特征对比）
├── url_security.py            # URL 安全检测（本地规则 + 风险评级）
├── tokenizer_utils.py         # jieba 分词器共享模块
├── generate_ai_phishing.py    # AI 钓鱼邮件训练数据批量生成
├── generate_ai_test.py        # AI 钓鱼邮件测试数据生成（消融专用）
├── integrate_chifraud.py      # CHIFRAUD 数据处理
├── app_gradio.py              # Gradio Web 界面
├── test.py                    # EML 查看工具
└── README.md
```


## 🚀 快速开始

### 环境要求

```
Python >= 3.10
CUDA >= 11.8
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
| 困惑度归一化 | `log1p(perplexity)` | 将 27K\~136K 压缩到 ~10\~12，避免 FP16 NaN |

#### 5. URL 安全检测 (`url_security.py`)

本地规则引擎，无需外部 API，对提取到的链接进行 10 项安全评级：

| 检测项 | 风险分 | 说明 |
|------|:---:|------|
| HTTP 非加密 | +1 | 钓鱼站 HTTPS 部署率远低于正常站 |
| IP 直连 | +3 | 正规服务使用域名，非 IP |
| 短网址服务 | +3 | bit.ly, t.cn 等 16 个短网址平台 |
| 可疑顶级域 | +2 | .tk, .ml, .xyz 等免费/滥用域名 |
| 仿冒品牌 | +2~3 | 域名含 apple、google 等品牌名但非官方域名 |
| 连字符过多 | +1 | 钓鱼域名常用 `-` 拼接 (apple-id-verify) |
| 钓鱼敏感词 | +1 | verify, secure, 验证, 安全 等 |
| 子域名过深 | +1 | >4 级子域名常用于混淆 |
| URL 含 @ 符号 | +4 | 经典重定向攻击手法 |
| 异常端口 | +1 | 非标准端口 |

**后处理加权**：URL 最高风险分 ≥6 → 模型概率 +0.15；≥4 → +0.08。将 URL 安全信号直接注入最终判定。

#### 6. AI 生成邮件检测策略

系统通过 **三层递进** 识别 AI 生成的钓鱼邮件：

```
┌─────────────────────────────────────────────┐
│ 第一层：统计层 (Perplexity)                  │
│ AI 文本 token 序列更"可预测"，BERT MLM 计算  │
│ 困惑度显著低于人类文本.取log归一化后作为      │
│ 第 9 维数值特征，同时送入 RF 和 BERT 融合层。 │
├─────────────────────────────────────────────┤
│ 第二层：语义层 (BERT 微调)                    │
│ 训练集中包含 5000 封 LLM 生成的钓鱼邮件，覆盖   │
│ 账号安全、福利补贴、快递物流、企业冒名、退款理赔 │
│ 5 种类型 × 2 种语气。BERT 解冻顶层学习 AI 文本  │
│ 的句法规整性、标点规范性、段落均匀性等模式。     │
├─────────────────────────────────────────────┤
│ 第三层：链接层 (URL 安全检测)                  │
│ AI 生成钓鱼邮件的链接通常也是伪造域名（仿冒品牌  │
│ + 连字符 + 钓鱼关键词），URL 安全模块独立评级    │
│ 并通过后处理加权修正最终概率。                   │
└─────────────────────────────────────────────┘
```

消融实验验证：在相同训练数据下，含 perplexity 特征的 BERT 在纯 AI 测试集上的 F1 优于不含版本（实验待补全）。

#### 7. 系统运行截图
下图是不同引擎对AI生成邮件的检测情况：

下图是对人类编写的钓鱼邮件的检测情况：


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
