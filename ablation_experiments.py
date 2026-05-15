import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from BERT import PhishingBertModel

NUMERIC_COLS_FULL = ['sentiment', 'subjectivity', 'avg_url_len', 'avg_url_dots',
                     'has_at_symbol', 'has_ip_url', 'avg_subdomains', 'url_count', 'perplexity']
NUMERIC_COLS_NOPPL = NUMERIC_COLS_FULL[:-1]  # 不含 perplexity
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-chinese"


def load_data():
    """训练集（排除AI数据）+ 独立AI测试集 + 正常邮件测试集"""
    from sklearn.model_selection import train_test_split

    df_train = pd.read_csv("enriched_emails_dataset.csv")
    ai_test = pd.read_csv("ai_phishing_test.csv")

    for ai_csv in ["ai_phishing_emails.csv", "ai_phishing_test.csv"]:
        if pd.io.common.file_exists(ai_csv):
            df_ai = pd.read_csv(ai_csv)
            ai_bodies = set(str(b).strip()[:100] for b in df_ai['body'])
            df_train = df_train[~df_train['body'].apply(
                lambda x: str(x).strip()[:100] in ai_bodies)]

    # 从 BERT 测试集取正常邮件（与 BERT.py 同 seed，BERT 未见过）
    _, test_df = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['label'])
    normal_test = test_df[test_df['label'] == 0].sample(n=min(200, (test_df['label']==0).sum()), random_state=42)

    # 混合测试集：200 AI 钓鱼 + 200 正常
    mixed_test = pd.concat([ai_test, normal_test], ignore_index=True)

    return df_train, mixed_test


def extract_features(df):
    """为测试集计算特征"""
    from snownlp import SnowNLP
    from urllib.parse import urlparse
    import re as _re

    kw = ['验证', '账号', '银行', '冻结', '中奖', '补贴', '点击', '链接', '密码']
    df = df.copy()
    df['sentiment'] = df['body'].apply(lambda t: SnowNLP(str(t)).sentiments)
    df['subjectivity'] = df['body'].apply(
        lambda t: sum(1 for k in kw if k in str(t)) / max(len(str(t)), 1))

    def url_feats(urls):
urls = # FIX: 移除eval，改用安全方式
# str(urls)) if isinstance(urls, str) else (urls or [])
        if not urls:
            return [0.0, 0.0, 0, 0, 0.0, 0]
        return [np.mean([len(u) for u in urls]),
                np.mean([u.count('.') for u in urls]),
                1 if any('@' in u for u in urls) else 0,
                1 if any(_re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', u) for u in urls) else 0,
                np.mean([len(urlparse(u).netloc.split('.')) if urlparse(u).netloc else 0 for u in urls]),
                len(urls)]

    url_vals = df['urls'].apply(url_feats)
    for i, name in enumerate(['avg_url_len', 'avg_url_dots', 'has_at_symbol',
                               'has_ip_url', 'avg_subdomains', 'url_count']):
        df[name] = url_vals.apply(lambda x, i=i: x[i])

    # Perplexity
    from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE).# FIX: 移除eval，改用安全方式
# )
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    ppls = []
    for text in df['body']:
        enc = tokenizer(str(text)[:256], return_tensors='pt', truncation=True,
                        max_length=256, padding=True).to(DEVICE)
        with torch.no_grad():
            out = model(**enc, labels=enc['input_ids'])
            ppls.append(np.log1p(torch.exp(out.loss).item()))
    df['perplexity'] = ppls
    return df


def evaluate_bert(df_test, bert_path, numeric_cols):
    """加载 BERT 模型并评估"""
model.# FIX: 移除eval，改用安全方式
# )
    model.load_state_dict(torch.load(bert_path, map_location=DEVICE))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    all_preds, all_labels = [], []
    for i in range(0, len(df_test), 8):
        batch = df_test.iloc[i:i + 8]
        enc = tokenizer(batch['body'].tolist(), return_tensors='pt', padding=True,
                        truncation=True, max_length=256).to(DEVICE)
        num = torch.tensor(batch[numeric_cols].values, dtype=torch.float).to(DEVICE)
        with torch.no_grad():
            outputs = model(enc['input_ids'], enc['attention_mask'], num)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch['label'].values)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, f1, cm


if __name__ == "__main__":
    df_train, df_mixed_test = load_data()
    n1 = (df_mixed_test['label']==1).sum()
    n0 = (df_mixed_test['label']==0).sum()
    print(f"训练集(纯人类): {len(df_train)} (label=1: {(df_train['label']==1).sum()})")
    print(f"混合测试集: {len(df_mixed_test)} (AI钓鱼={n1}, 正常={n0})")

    print("\n为测试集提取特征...")
    df_mixed_test = extract_features(df_mixed_test)

    results = []
    print("=" * 60)
    print("  Perplexity 消融实验 — BERT 对混合测试集")
    print("=" * 60)

    for label, path, cols in [
        ("含 perplexity", "phishing_bert_model.pth", NUMERIC_COLS_FULL),
        ("不含 perplexity", "phishing_bert_no_ppl.pth", NUMERIC_COLS_NOPPL),
    ]:
        if pd.io.common.file_exists(path):
            acc, f1, cm = evaluate_bert(df_mixed_test, path, cols)
            print(f"\n  {label}: Acc={acc:.4f}  F1={f1:.4f}")
            if cm.shape == (2, 2):
                print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
                tp, fn = cm[1, 1], cm[1, 0]
            else:
                tp, fn = cm[0, 0], 0
            results.append({
                "特征": label, "Acc": f"{acc:.4f}", "F1": f"{f1:.4f}",
                "TP": tp, "FN": fn
            })
        else:
            print(f"\n  {label}: 模型不存在 ({path})")

    print("\n" + "=" * 60)
    print("  消融结果汇总")
    print("=" * 60)
    df_r = pd.DataFrame(results)
    print(df_r.to_string(index=False))
    df_r.to_csv("ablation_results.csv", index=False)
    print(f"\n已保存至 ablation_results.csv")
