import jieba


def jieba_tokenizer(text):
    """jieba 分词器，供RF.py和predict.py共用"""
    return jieba.lcut(text)
