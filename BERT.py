import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 1. 参数配置
DEVICE = torch.cuda.is_available() and "cuda" or "cpu"
MODEL_NAME = "bert-base-chinese"
MAX_LEN = 256
BATCH_SIZE = 4
EPOCHS = 3
BERT_LR = 2e-5
HEAD_LR = 1e-4
UNFREEZE_TOP_N = 4

NUMERIC_COLS = [
    'sentiment', 'subjectivity', 'avg_url_len', 'avg_url_dots',
    'has_at_symbol', 'has_ip_url', 'avg_subdomains', 'url_count',
    'perplexity'
]


# 2. 自定义数据集类
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


# 3. 混合模型 (BERT + MLP)
class PhishingBertModel(nn.Module):
    def __init__(self, n_numeric_feats, unfreeze_top_n=UNFREEZE_TOP_N):
        super(PhishingBertModel, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)

        # 冻结全部参数
        for param in self.bert.parameters():
            param.requires_grad = False

        # 解冻顶层 N 层
        layers = self.bert.encoder.layer

        num_layers = len(layers)
        for i in range(num_layers - unfreeze_top_n, num_layers):
            for param in layers[i].parameters():
                param.requires_grad = True

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(768 + n_numeric_feats, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, numeric_feats):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        combined = torch.cat((pooled_output, numeric_feats), dim=1)
        return self.classifier(combined)


# 4. 训练逻辑
def train_model():
    df = pd.read_csv("enriched_emails_dataset.csv")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_set = EmailDataset(
        train_df['body'].values, train_df[NUMERIC_COLS].values,
        train_df['label'].values, tokenizer
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    model = PhishingBertModel(len(NUMERIC_COLS)).to(DEVICE)

    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': BERT_LR},
        {'params': model.classifier.parameters(), 'lr': HEAD_LR}
    ])
    criterion = nn.CrossEntropyLoss()

    trainable_bert = sum(p.numel() for p in model.bert.parameters() if p.requires_grad)
    total_bert = sum(p.numel() for p in model.bert.parameters())
    trainable_head = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    print(f"模型: {MODEL_NAME}")
    print(f"BERT 可训练参数: {trainable_bert:,} / {total_bert:,} ({trainable_bert/total_bert*100:.1f}%)")
    print(f"分类头可训练参数: {trainable_head:,}")
    print(f"BERT 骨干学习率: {BERT_LR}, 分类头学习率: {HEAD_LR}")

    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"开始训练... 使用设备: {DEVICE} (混合精度: {'ON' if use_amp else 'OFF'})")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, batch in enumerate(progress):
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(
                    input_ids=batch['input_ids'].to(DEVICE),
                    attention_mask=batch['attention_mask'].to(DEVICE),
                    numeric_feats=batch['numeric_feats'].to(DEVICE)
                )
                loss = criterion(outputs, batch['labels'].to(DEVICE))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            if (batch_idx + 1) % 50 == 0:
                progress.write(f"  batch {batch_idx+1}/{len(train_loader)} | loss={loss.item():.4f} | avg_loss={total_loss/(batch_idx+1):.4f}")
        print(f"  -> Epoch {epoch+1}/{EPOCHS} Avg Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), 'phishing_bert_model.pth')
    print("模型已保存！")


if __name__ == "__main__":
    train_model()
