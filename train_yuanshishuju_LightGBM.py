import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import lightgbm as lgb  # 引入 LightGBM
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os
import shap


# ==========================================
# 0. 中文显示修复
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
# 1. 加载数据
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

# ==========================================
# 3. 训练 LightGBM 模型
# ==========================================
model = None
le = None
y_test = None
y_pred = None
X_test = None

if len(df_ml) > 0:
    print("\n🚀 正在训练 LightGBM 模型...")
    X = df_ml[feature_cols]
    y = df_ml['Final_Label']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 计算类别权重 (LightGBM 处理不平衡数据的方式)
    class_counts = np.bincount(y_encoded)
    total_samples = len(y_encoded)
    n_classes = len(class_counts)
    # 计算 scale_pos_weight 类似的权重，这里用 is_unbalance 或手动权重
    # 为了简单且有效，我们使用 is_unbalance=True 参数，或者手动计算 sample_weight

    # 切分数据集
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # LightGBM 模型定义
    # 注意：LightGBM 的默认参数通常很强，这里先使用默认配置
    # is_unbalance=True 会自动调整权重，适合你的类别不平衡情况
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=4,
        n_estimators=200,  # 树的数量
        learning_rate=0.05,  # 学习率稍微调小一点，更稳健
        max_depth=-1,  # 不限制深度，靠 num_leaves 控制
        num_leaves=31,  # 叶子节点数
        subsample=0.8,  # 样本采样
        colsample_bytree=0.8,  # 特征采样
        is_unbalance=True,  # 自动处理类别不平衡
        random_state=42,
        verbose=-1,  # 关闭训练日志
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n✅ LightGBM 训练完成！分类报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(model, 'ml_classifier_lightgbm.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("💾 模型已保存")

    # ==========================================
    # 4. 全套可视化分析
    # ==========================================
    print("\n📊 正在生成可视化图表...")
    target_names = le.classes_

    # --- 图1：混淆矩阵 ---
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('混淆矩阵：LightGBM 分类分析', fontsize=16)
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.tight_layout()
    plt.savefig('1_混淆矩阵_LGBM.png', dpi=300)
    plt.show()

    # --- 图2：特征重要性 ---
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(model, max_num_features=10, height=0.8, importance_type='gain')
    plt.title('特征重要性排序 (LightGBM)', fontsize=16)
    plt.xlabel('重要性得分 (Gain)', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    plt.tight_layout()
    plt.savefig('2_特征重要性_LGBM.png', dpi=300)
    plt.show()

    # --- 图3：雷达图 ---
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    angles = np.linspace(0, 2 * np.pi, len(target_names), endpoint=False).tolist()
    angles += angles[:1]
    p_list = p.tolist();
    p_list += p_list[:1]
    r_list = r.tolist();
    r_list += r_list[:1]
    f1_list = f1.tolist();
    f1_list += f1_list[:1]
    ax.plot(angles, p_list, 'o-', linewidth=2, label='精确率')
    ax.fill(angles, p_list, alpha=0.25)
    ax.plot(angles, r_list, 's-', linewidth=2, label='召回率')
    ax.fill(angles, r_list, alpha=0.25)
    ax.plot(angles, f1_list, '^-', linewidth=2, label='F1分数')
    ax.fill(angles, f1_list, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(target_names, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title('各类别性能评估雷达图', fontsize=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig('3_性能评估雷达图_LGBM.png', dpi=300)
    plt.show()

    # --- 图4：饼图 ---
    plt.figure(figsize=(10, 7))
    counts = df_ml['Final_Label'].value_counts()
    colors = sns.color_palette('pastel')
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=colors,
            explode=[0.05 if i == 'Class_3_大幅度运动' else 0 for i in counts.index])
    plt.title('机器学习数据集类别分布', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('4_数据分布饼图_LGBM.png', dpi=300)
    plt.show()

    # --- 图5：箱线图 ---
    plt.figure(figsize=(12, 6))
    y_test_labels = le.inverse_transform(y_test)
    df_viz = pd.DataFrame({'max_val': X_test['max_val'].values, 'label': y_test_labels})
    sns.boxplot(x='label', y='max_val', data=df_viz, palette='Set2')
    plt.title('各类别在 "最大值(max_val)" 上的分布对比', fontsize=16)
    plt.ylabel('最大值', fontsize=12)
    plt.xlabel('类别', fontsize=12)
    plt.tight_layout()
    plt.savefig('5_关键特征分布箱线图_LGBM.png', dpi=300)
    plt.show()

    # ==========================================
    # 5. SHAP 可解释性分析 (LightGBM 完美支持)
    # ==========================================
    print("\n🔍 正在启动 SHAP 可解释性分析...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # --- 图6：SHAP 摘要图 ---
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False, plot_type="bar")
    plt.title('全局特征重要性 (基于SHAP值)', fontsize=16)
    plt.tight_layout()
    plt.savefig('6_SHAP_全局重要性_LGBM.png', dpi=300)
    plt.show()

    # --- 图7：SHAP 依赖图 ---
    plt.figure(figsize=(12, 8))
    shap.dependence_plot("max_val", shap_values, X_test, feature_names=feature_cols, show=False)
    plt.title('特征 "max_val" 的 SHAP 依赖图', fontsize=16)
    plt.tight_layout()
    plt.savefig('7_SHAP_max_val_依赖图_LGBM.png', dpi=300)
    plt.show()

    print("✅ SHAP 分析图表已生成！")

else:
    print("⚠️ 警告：没有数据留给机器学习训练，跳过绘图。")

# ==========================================
# 6. 最终混合验证
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
            method = "LightGBM"
        results.append((row['Final_Label'], final_pred, method))

    print(f"{'真实标签':<20} | {'预测结果':<20} | {'判定方式'}")
    print("-" * 60)
    for real, pred, method in results:
        status = "✅" if real == pred else "❌"
        print(f"{status} {real:<18} | {pred:<18} | {method}")

print("\n🎉 流程结束！请查看生成的图片文件。")