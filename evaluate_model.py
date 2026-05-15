import torch
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from tokenizer_utils import jieba_tokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from BERT import PhishingBertModel, EmailDataset

# 1. 配置
DEVICE = torch.cuda.is_available() and "cuda" or "cpu"
BERT_MODEL_PATH = 'phishing_bert_model.pth'
DATA_PATH = 'enriched_emails_dataset.csv'
MODEL_NAME = "bert-base-chinese"
N_FOLDS = 5

NUMERIC_COLS = [
    'sentiment', 'subjectivity', 'avg_url_len', 'avg_url_dots',
    'has_at_symbol', 'has_ip_url', 'avg_subdomains', 'url_count',
    'perplexity'
]

def evaluate_rf_cv(df):
    """5 折交叉验证：每折分别训练+评估 RF"""
    print("\n--- 随机森林 5 折交叉验证 ---")
    X = df[['body'] + NUMERIC_COLS]
    y = df['label']

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_acc, fold_f1 = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        preprocessor = ColumnTransformer([
            ('text', TfidfVectorizer(max_features=500, tokenizer=jieba_tokenizer), 'body'),
            ('num', 'passthrough', NUMERIC_COLS)
        ])
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fold_acc.append(acc)
        fold_f1.append(f1)
        print(f"  Fold {fold+1}: Acc={acc:.4f}, F1={f1:.4f}")

    print(f"  Mean: Acc={np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}, F1={np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")
    return np.mean(fold_acc), np.std(fold_acc), np.mean(fold_f1), np.std(fold_f1)

def evaluate_bert_cv(df):
    """加载已训练 BERT 模型，用 5 次不同随机切分评估泛化稳定性"""
    print("\n--- DistilBERT 5 次随机切分评估 ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = PhishingBertModel(len(NUMERIC_COLS)).to(DEVICE)
    model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=DEVICE))
    model.eval()

    fold_acc, fold_f1 = [], []

    for seed in range(N_FOLDS):
        test_df = df.sample(frac=0.2, random_state=seed + 100)

        test_set = EmailDataset(
            test_df['body'].values, test_df[NUMERIC_COLS].values,
            test_df['label'].values, tokenizer
        )
        test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(
                    input_ids=batch['input_ids'].to(DEVICE),
                    attention_mask=batch['attention_mask'].to(DEVICE),
                    numeric_feats=batch['numeric_feats'].to(DEVICE)
                )
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['labels'].numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        fold_acc.append(acc)
        fold_f1.append(f1)
        print(f"  Seed {seed+100}: Acc={acc:.4f}, F1={f1:.4f} (test={len(test_df)})")

    print(f"  Mean: Acc={np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}, F1={np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")
    return np.mean(fold_acc), np.std(fold_acc), np.mean(fold_f1), np.std(fold_f1)


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    print(f"数据总量: {len(df)}, label=0: {(df['label']==0).sum()}, label=1: {(df['label']==1).sum()}")

    rf_acc, rf_acc_std, rf_f1, rf_f1_std = evaluate_rf_cv(df)
    bert_acc, bert_acc_std, bert_f1, bert_f1_std = evaluate_bert_cv(df)

    print("\n" + "=" * 55)
    print("           模型 5 折交叉验证对比")
    print("=" * 55)
    results = pd.DataFrame({
        "指标": ["Accuracy", "F1-Score"],
        "Random Forest": [f"{rf_acc:.4f} ± {rf_acc_std:.4f}", f"{rf_f1:.4f} ± {rf_f1_std:.4f}"],
        "DistilBERT":   [f"{bert_acc:.4f} ± {bert_acc_std:.4f}", f"{bert_f1:.4f} ± {bert_f1_std:.4f}"],
    })
    print(results.to_string(index=False))
