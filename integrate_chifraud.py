import pandas as pd
import re
import ast

DATA_DIR = "chifraud_data"


def clean_text(text):
    if pd.isna(text) or not str(text).strip():
        return "Empty_Body"
    from bs4 import BeautifulSoup
    text = str(text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else "Empty_Body"


def load_chifraud_csv(filepath, label_map):
    """读入 CHIFRAUD TSV，列对齐到 pipeline 格式"""
    df = pd.read_csv(filepath, sep='\t')
    df.columns = ['label_id', 'body']

    # 二分类：0=正常, 1-10=欺诈
    df['label'] = df['label_id'].apply(lambda x: 0 if x == 0 else 1)

    df['body'] = df['body'].apply(clean_text)
    df['subject'] = 'Unknown'
    df['sender'] = 'Unknown'
    df['reply_to'] = 'None'
    df['received_count'] = 0
    df['urls'] = [[] for _ in range(len(df))]
    df['url_count'] = 0
    df['source_dataset'] = filepath

    return df[['subject', 'sender', 'reply_to', 'received_count',
               'body', 'urls', 'url_count', 'label', 'source_dataset']]


if __name__ == "__main__":
    print("Integrating CHIFRAUD datasets...")
    df_train = load_chifraud_csv(f"{DATA_DIR}/ChiFraud_train.csv", None)
    df_2022 = load_chifraud_csv(f"{DATA_DIR}/ChiFraud_t2022.csv", None)
    df_2023 = load_chifraud_csv(f"{DATA_DIR}/ChiFraud_t2023.csv", None)

    # 合并并全局去重
    df_all = pd.concat([df_train, df_2022, df_2023], ignore_index=True)
    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=['body'], keep='first')
    after = len(df_all)

    print(f"  Train: {len(df_train)}")
    print(f"  2022:  {len(df_2022)}")
    print(f"  2023:  {len(df_2023)}")
    print(f"  Total: {before} → {after} (dedup removed {before - after})")
    print(f"  label=0 (normal):  {(df_all['label'] == 0).sum()}")
    print(f"  label=1 (fraud):   {(df_all['label'] == 1).sum()}")

    out_path = "chifraud_emails.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
