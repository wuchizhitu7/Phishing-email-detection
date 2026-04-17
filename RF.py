import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. 加载数据
df = pd.read_csv("enriched_emails_dataset.csv")

# 2. 定义特征逻辑
numeric_features = [
    'sentiment', 'subjectivity', 'avg_url_len',
    'avg_url_dots', 'has_at_symbol', 'has_ip_url',
    'avg_subdomains', 'url_count'
]
text_feature = 'body'

# 3. 构建混合预处理器
# 对正文进行 TF-IDF 处理（提取前 500 个关键词），对数值特征保持不变
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=500, stop_words='english'), text_feature),
        ('num', 'passthrough', numeric_features)
    ])

# 4. 组装流水线
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])

# 5. 准备数据
X = df[[text_feature] + numeric_features]
y = df['label']

# 6. 5折交叉验证，验证真实泛化能力
print("正在进行交叉验证...")
cv_scores = cross_val_score(pipeline, X, y, cv=5)
print(f"5折交叉验证平均得分: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 7. 训练模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n--- 剔除泄露特征后的评估报告 ---")
print(classification_report(y_test, y_pred))

# 8. 查看新的特征重要性
# 获取文本特征的名称
tfidf_features = pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()
all_feature_names = list(tfidf_features) + numeric_features
importances = pipeline.named_steps['classifier'].feature_importances_

importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
print("\n--- 贡献度前 10 的特征 ---")
print(importance_df.sort_values(by='importance', ascending=False).head(10))

# 保存模型
model_filename = 'phishing_detector_final.pkl'
joblib.dump(pipeline, model_filename)

print(f"\n[成功] 最终模型已保存为: {model_filename}")

# 保存特征列名，方便预测时对齐
joblib.dump(numeric_features, 'numeric_features_list.pkl')
