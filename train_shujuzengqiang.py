import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置与加载数据
# ==========================================
print("⚡ 正在加载数据...")
# 请确保文件名正确
df = pd.read_excel('最终合并数据集_原始数据.xlsx')

# 定义特征列
feature_cols = ['max_val', 'mean_val', 'sum_val', 'std_val', 'width', 'center', 'span']

# 数据清洗：确保特征列是数值型
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=feature_cols)

print(f"原始数据总量: {len(df)}")


# ==========================================
# 2. 第一层：物理规则拦截 Class_4_跌倒
# ==========================================
def physics_fall_filter(row):
    """
    专门针对 Class_4_跌倒 的物理规则
    逻辑：宽覆盖 (width > 40) 且 低能量 (max_val < 8000)
    返回 True 表示是跌倒，False 表示交给模型
    """
    max_val = row['max_val']
    width = row['width']

    # 这里的阈值基于你之前的逻辑，可根据实际情况微调
    if width > 40 and max_val < 8000:
        return True
    return False


# 应用规则，将数据分为两部分
# mask_fall 是布尔值，True 代表被规则判定为跌倒
mask_fall = df.apply(physics_fall_filter, axis=1)

# 1. 跌倒数据 (直接由规则判定，不参与训练)
df_fall = df[mask_fall]

# 2. 机器学习数据 (剩下的所有数据：0, 1, 2, 3)
df_ml = df[~mask_fall].copy()

print(f"🚨 物理规则拦截跌倒样本数: {len(df_fall)}")
print(f"🧠 机器学习处理样本数: {len(df_ml)}")
print(f"机器学习数据分布:\n{df_ml['Final_Label'].value_counts()}")

# ==========================================
# 3. 第二层：训练 XGBoost 模型 (4分类)
# ==========================================
if len(df_ml) > 0:
    print("\n🚀 正在训练 XGBoost 模型 (4分类: 空置/静止/正常/大幅度)...")

    # 准备特征和标签
    X = df_ml[feature_cols]
    y = df_ml['Final_Label']

    # 标签编码 (将文字转为数字 0, 1, 2, 3)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 划分训练集和测试集
    # 注意：这里不再 stratify Class_4，因为它已经被移除了
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 定义模型
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,  # 4分类
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        random_state=42,
        eval_metric='mlogloss'
    )

    # 训练
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)

    print("\n✅ 训练完成！机器学习部分 (Class 0/1/2/3) 分类报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 绘制混淆矩阵 (可选)
    # plt.figure(figsize=(8,6))
    # sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    # plt.title('机器学习部分混淆矩阵')
    # plt.show()

    # 保存模型
    joblib.dump(model, 'ml_classifier_4class.pkl')
    joblib.dump(le, 'label_encoder_4class.pkl')
    print("💾 模型已保存为 'ml_classifier_4class.pkl'")

else:
    print("⚠️ 警告：没有数据留给机器学习训练！")

# ==========================================
# 4. 最终混合验证
# ==========================================
print("\n--- 🧪 混合系统整体验证 ---")

# 随机抽取 10 条数据进行演示
sample_indices = np.random.choice(len(df), 10, replace=False)
samples = df.iloc[sample_indices]

results = []

for i, row in samples.iterrows():
    # 第一步：先过物理规则
    if physics_fall_filter(row):
        final_pred = "Class_4_跌倒"
        method = "物理规则"
    else:
        # 第二步：再过机器学习模型
        if len(df_ml) > 0:
            x_input = row[feature_cols].values.reshape(1, -1)
            pred_code = model.predict(x_input)[0]
            final_pred = le.inverse_transform([pred_code])[0]
            method = "机器学习"
        else:
            final_pred = "N/A"
            method = "无模型"

    results.append((row['Final_Label'], final_pred, method))

# 打印结果表格
print(f"{'真实标签':<20} | {'预测结果':<20} | {'判定方式'}")
print("-" * 60)
for real, pred, method in results:
    status = "✅" if real == pred else "❌"
    print(f"{status} {real:<18} | {pred:<18} | {method}")

print("\n🎉 流程结束！")