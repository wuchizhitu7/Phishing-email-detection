import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- 1. 参数配置 ---
DEVICE = torch.cuda.is_available() and "cuda" or "cpu"
MODEL_NAME = "distilbert"
MAX_LEN = 512  # BERT 最大长度
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5


# --- 2. 自定义数据集类 ---
class EmailDataset(Dataset):
    def __init__(self, texts, numeric_feats, labels, tokenizer):
        self.texts = texts
        self.numeric_feats = torch.tensor(numeric_feats, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

       # truncation=True 会自动处理超过 512 的部分
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'numeric_feats': self.numeric_feats[idx],
            'labels': self.labels[idx]
        }


# --- 3. 混合模型架构 (BERT + MLP) ---
class PhishingBertModel(nn.Module):
    def __init__(self, n_numeric_feats):
        super(PhishingBertModel, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        # 冻结 BERT 前几层（可选，为了在 CPU 上训练更快）
        for param in self.bert.parameters(): param.requires_grad = False

        self.dropout = nn.Dropout(0.3)
        # 融合层：BERT 的 768 维 + 你的 8 维数值特征
        self.classifier = nn.Sequential(
            nn.Linear(768 + n_numeric_feats, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # 0: 正常, 1: 钓鱼
        )

    def forward(self, input_ids, attention_mask, numeric_feats):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] 向量
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # 特征拼接 (Feature Fusion)
        combined = torch.cat((pooled_output, numeric_feats), dim=1)
        return self.classifier(combined)


# --- 4. 训练逻辑 ---
def train_model():
    # 加载数据
    df = pd.read_csv("enriched_emails_dataset.csv")
    numeric_cols = ['sentiment', 'subjectivity', 'avg_url_len', 'avg_url_dots',
                    'has_at_symbol', 'has_ip_url', 'avg_subdomains', 'url_count']

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_set = EmailDataset(train_df['body'].values, train_df[numeric_cols].values, train_df['label'].values,
                             tokenizer)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    model = PhishingBertModel(len(numeric_cols)).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"开始训练... 使用设备: {DEVICE}")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'].to(DEVICE),
                attention_mask=batch['attention_mask'].to(DEVICE),
                numeric_feats=batch['numeric_feats'].to(DEVICE)
            )
            loss = criterion(outputs, batch['labels'].to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")

    # 保存模型状态
    torch.save(model.state_dict(), 'phishing_bert_model.pth')
    print("模型已保存！")


if __name__ == "__main__":
    train_model()