# 🛡️ 中文钓鱼邮件智能检测系统

> 基于 **Random Forest + BERT 双引擎混合架构**，融入 **Perplexity 困惑度特征**，具备 AI 生成钓鱼邮件检测能力。

## 演示视频


https://github.com/user-attachments/assets/ea843917-8141-4205-80f5-c864bf3607b9



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

<img width="1448" height="1086" alt="流程图" src="https://github.com/user-attachments/assets/addda121-ec41-4773-af18-880578dcc70f" />

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

##### Perplexity 检测 AI 文本

大型语言模型在生成文本时，本质上是逐 token 做概率采样——每一步都选择当前上下文下"最合理"的下一个 token。这个过程使得 AI 生成的文本具有一个统计特性：**token 序列比人类写作更"可预测"**。

BERT MLM（掩码语言模型）在计算困惑度时，会遍历文本的每个 token，用上下文预测该位置最可能的词，然后计算交叉熵损失。AI 文本由于 token 之间的过渡更平滑、更符合语言模型的预期分布，困惑度**偏低**；人类写作因用词多样、句式跳跃、偶有错别字/不规范表达，困惑度**偏高**。

| 文本类型 | 困惑度 | 原因 |
|------|:---:|------|
| AI 生成钓鱼邮件 | **低** | token 序列平滑，用词集中在高频范围，句法高度规整 |
| 人类正常商务邮件 | 中等 | 用词丰富但格式规范，有一定不可预测性 |
| 旧垃圾邮件（TREC06C） | **高** | 编码损坏、乱码、拼写错误、不规则空格 |

##### Perplexity 的实际影响

| 维度 | 效果 |
|------|------|
| **对 RF** | 作为第 9 维数值特征送入，AI 钓鱼与正常邮件在困惑度轴上产生分离度 |
| **对 BERT** | 与 [CLS] 语义向量拼接后送入融合层，为语义理解提供统计维度的辅助信号 |
| **单独可靠性** | 不足以独立判定——格式规范的人类邮件同样低困惑度，会误杀 |
| **组合价值** | 当 BERT 判为钓鱼 **且** perplexity 极低时，强烈指向 AI 生成来源 |

##### 技术处理

原始 perplexity 值域为 27K~136K，在 FP16 训练下会导致 NaN。系统通过 `log1p` 变换将其压缩到 ~10~12 范围，解决数值稳定性问题。

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

| 层级 | 检测方式 | 核心原理 |
|:---:|---|---|
| **第1层 · 统计层** | Perplexity 困惑度 | AI 文本 token 序列平滑可预测，BERT MLM 计算困惑度显著低于人类写作。取 log 归一化后作为第 9 维数值特征，同时送入 RF 和 BERT 融合层。 |
| **第2层 · 语义层** | BERT 微调 | 训练集含 5,000 封 LLM 生成钓鱼邮件，覆盖账号安全、福利补贴、快递物流、企业冒名、退款理赔 5 大类型，每种含紧急胁迫 / 温和商务两种语气。BERT 解冻顶层 4 层，学习 AI 文本特有话术组合与篇章结构。 |
| **第3层 · 链接层** | URL 安全检测 | AI 生成邮件的链接同样为伪造域名（仿冒品牌 + 连字符 + 钓鱼敏感词）。URL 安全模块对链接进行 10 项本地规则评级，高风险链接通过后处理加权 (+0.08~0.15) 修正最终概率。 |


#### 7. 系统运行截图
下图是不同引擎对AI生成邮件的检测情况：
<img width="1850" height="546" alt="image" src="https://github.com/user-attachments/assets/31c465a3-f588-436a-ae50-c1d6ea66ed8d" />
<img width="1786" height="641" alt="image" src="https://github.com/user-attachments/assets/44130b71-27e8-47ff-a45f-3856dc8effa4" />
<img width="1853" height="596" alt="image" src="https://github.com/user-attachments/assets/02c503b6-2076-45d9-806a-30f02e4f15ef" />

下图是对人为编写的钓鱼邮件的检测情况：
<img width="1798" height="813" alt="image" src="https://github.com/user-attachments/assets/78b5f40d-48a0-439a-9089-bc460bba2a62" />
<img width="1712" height="717" alt="image" src="https://github.com/user-attachments/assets/e422f9d8-678a-4046-a931-4db89abcf720" />
<img width="1790" height="722" alt="image" src="https://github.com/user-attachments/assets/ce33bb73-a77c-4dca-a43c-54193b9223f1" />

## 📊 评估指标

5 折分层交叉验证 — 数据量 84,482 条（正常 49,513 / 欺诈 34,969）：

| 指标 | Random Forest | bert-base-chinese |
|------|:---:|:---:|
| **Accuracy** | 0.9782 ± 0.0005 | **0.9967 ± 0.0005** |
| **F1-Score** | 0.9735 ± 0.0006 | **0.9960 ± 0.0006** |
| 稳定性 (σ) | ±0.0006 | **±0.0006** |
| 训练时间 | ~3 分钟 | ~2 小时 (GPU) |


## 📦 数据来源

| 数据集 | 年份 | 规模（使用） | 用途 |
|------|:---:|:---:|------|
| [TREC06C](https://plg.uwaterloo.ca/cgi-bin/cgiwrap/gvcormac/foo06) | 2006 | 19,500 条 | 正常中文邮件（仅保留 ham） |
| [CHIFRAUD](https://github.com/xuemingxxx/ChiFraud) | 2025 | 各 30,000 条 | 正常 + 欺诈文本采样 |
| AI 生成钓鱼 | 2025 | 4,969 条 | 5 种钓鱼类型 × 多种语气 |


## 📄 License

MIT License
