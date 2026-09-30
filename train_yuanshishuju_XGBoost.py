import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import xgboost as xgb
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os


# ==========================================
# 0. 终极中文显示修复方案 (暴力加载字体)
# ==========================================
def fix_chinese_display():
    font_candidates = [
        "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"
    ]
    chosen_font = None
    for font_path in font_candidates:
        if os.path.exists(font_path):
            chosen_font = font_path
            break

    if chosen_font:
        my_font = fm.FontProperties(fname=chosen_font)
        plt.rcParams['font.family'] = my_font.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        print(f"✅ 字体修复成功：已加载字体 {chosen_font}")
    else:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("⚠️ 警告：未找到标准中文字体文件，尝试使用默认字体。")


fix_chinese_display()
warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置与加载数据
# ==========================================
print("⚡ 正在加载数据...")
try:
    df = pd.read_excel('最终合并数据集_原始数据.xlsx')
except FileNotFoundError:
    print("❌ 错误：找不到文件 '最终合并数据集_原始数据.xlsx'")
    exit()

feature_cols = ['max_val', 'mean_val', 'sum_val', 'std_val', 'width', 'center', 'span']

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=feature_cols)

print(f"原始数据总量: {len(df)}")


# ==========================================
# 2. 物理规则拦截 Class_4_跌倒
# ==========================================
def physics_fall_filter(row):
    max_val = row['max_val']
    width = row['width']
    if width > 40 and max_val < 8000:
        return True
    return False


mask_fall = df.apply(physics_fall_filter, axis=1)
df_fall = df[mask_fall]
df_ml = df[~mask_fall].copy()
df_ml = df_ml[df_ml['Final_Label'] != 'Class_4_跌倒']

print(f"🚨 物理规则拦截跌倒样本数: {len(df_fall)}")
print(f"🧠 机器学习处理样本数: {len(df_ml)}")
print("机器学习数据分布:")
print(df_ml['Final_Label'].value_counts())

# ==========================================
# 3. 训练 XGBoost 模型
# ==========================================
model = None
le = None
y_test = None
y_pred = None
X_test = None

if len(df_ml) > 0:
    print("\n🚀 正在训练 XGBoost 模型...")
    X = df_ml[feature_cols]
    y = df_ml['Final_Label']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 计算权重
    class_counts = np.bincount(y_encoded)
    total_samples = len(y_encoded)
    n_classes = len(class_counts)
    class_weights = {i: total_samples / (n_classes * count) for i, count in enumerate(class_counts)}
    sample_weights = np.array([class_weights[label] for label in y_encoded])

    # 切分
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    train_weights = sample_weights[train_idx]

    # 训练
    model = xgb.XGBClassifier(objective='multi:softprob', num_class=4, max_depth=6, learning_rate=0.1, n_estimators=200,
                              random_state=42, eval_metric='mlogloss')
    model.fit(X_train, y_train, sample_weight=train_weights)
    y_pred = model.predict(X_test)

    print("\n✅ 训练完成！分类报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(model, 'ml_classifier_final.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("💾 模型已保存")

    # ==========================================
    # 4. 全套可视化分析 (5张图)
    # ==========================================
    print("\n📊 正在生成并保存 5 张可视化图表...")
    target_names = le.classes_

    # --- 图1：混淆矩阵 ---
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('混淆矩阵：模型分类错误分析', fontsize=16)
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.tight_layout()
    plt.savefig('1_混淆矩阵.png', dpi=300)
    plt.show()

    # --- 图2：特征重要性 ---
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10, height=0.8, importance_type='weight')
    plt.title('特征重要性排序', fontsize=16)
    plt.xlabel('重要性得分 (Weight)', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    plt.tight_layout()
    plt.savefig('2_特征重要性.png', dpi=300)
    plt.show()

    # --- 图3：分类评估雷达图 (Precision/Recall) ---
    # 计算 Precision, Recall, F1
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    # 设置雷达图的角度
    angles = np.linspace(0, 2 * np.pi, len(target_names), endpoint=False).tolist()
    angles += angles[:1]  # 闭合圆环

    # 准备数据
    p_list = p.tolist();
    p_list += p_list[:1]
    r_list = r.tolist();
    r_list += r_list[:1]
    f1_list = f1.tolist();
    f1_list += f1_list[:1]

    # 绘制 Precision
    ax.plot(angles, p_list, 'o-', linewidth=2, label='精确率 (Precision)')
    ax.fill(angles, p_list, alpha=0.25)

    # 绘制 Recall
    ax.plot(angles, r_list, 's-', linewidth=2, label='召回率 (Recall)')
    ax.fill(angles, r_list, alpha=0.25)

    # 绘制 F1-Score
    ax.plot(angles, f1_list, '^-', linewidth=2, label='F1分数 (F1-Score)')
    ax.fill(angles, f1_list, alpha=0.25)

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(target_names, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title('各类别性能评估雷达图', fontsize=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig('3_性能评估雷达图.png', dpi=300)
    plt.show()

    # --- 图4：数据分布饼图 ---
    plt.figure(figsize=(10, 7))
    # 统计原始数据的分布
    counts = df_ml['Final_Label'].value_counts()
    colors = sns.color_palette('pastel')

    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=colors,
            explode=[0.05 if i == 'Class_3_大幅度运动' else 0 for i in counts.index])
    plt.title('机器学习数据集类别分布 (训练前)', fontsize=16)
    plt.axis('equal')  # 保证是正圆
    plt.tight_layout()
    plt.savefig('4_数据分布饼图.png', dpi=300)
    plt.show()

    # --- 图5：关键特征箱线图 (以 max_val 为例) ---
    plt.figure(figsize=(12, 6))
    # 将测试集的 y 转回标签
    y_test_labels = le.inverse_transform(y_test)
    df_viz = pd.DataFrame({
        'max_val': X_test['max_val'].values,
        'label': y_test_labels
    })

    sns.boxplot(x='label', y='max_val', data=df_viz, palette='Set2')
    plt.title('各类别在 "最大值(max_val)" 上的分布对比 (测试集)', fontsize=16)
    plt.ylabel('最大值 (信号强度)', fontsize=12)
    plt.xlabel('类别', fontsize=12)
    plt.tight_layout()
    plt.savefig('5_关键特征分布箱线图.png', dpi=300)
    plt.show()

else:
    print("⚠️ 警告：没有数据留给机器学习训练，跳过绘图。")

# ==========================================
# 5. 最终混合验证
# ==========================================
print("\n--- 🧪 混合系统整体验证 ---")
if model is not None:
    sample_indices = np.random.choice(len(df), 10, replace=False)
    samples = df.iloc[sample_indices]
    results = []

    for i, row in samples.iterrows():
        if physics_fall_filter(row):
            final_pred = "Class_4_跌倒"
            method = "物理规则"
        else:
            x_input = row[feature_cols].values.reshape(1, -1)
            pred_code = model.predict(x_input)[0]
            final_pred = le.inverse_transform([pred_code])[0]
            method = "机器学习"
        results.append((row['Final_Label'], final_pred, method))

    print(f"{'真实标签':<20} | {'预测结果':<20} | {'判定方式'}")
    print("-" * 60)
    for real, pred, method in results:
        status = "✅" if real == pred else "❌"
        print(f"{status} {real:<18} | {pred:<18} | {method}")

print("\n🎉 流程结束！请查看生成的图片文件。")