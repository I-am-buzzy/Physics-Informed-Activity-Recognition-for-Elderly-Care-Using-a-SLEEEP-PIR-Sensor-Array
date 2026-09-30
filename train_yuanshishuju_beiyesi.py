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

# 引入贝叶斯优化库
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score


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
# 3. 贝叶斯优化超参数搜索
# ==========================================
model = None
le = None
y_test = None
y_pred = None
X_test = None

if len(df_ml) > 0:
    print("\n🚀 正在准备数据...")
    X = df_ml[feature_cols]
    y = df_ml['Final_Label']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 计算类别权重
    class_counts = np.bincount(y_encoded)
    total_samples = len(y_encoded)
    n_classes = len(class_counts)
    class_weights = {i: total_samples / (n_classes * count) for i, count in enumerate(class_counts)}
    sample_weights = np.array([class_weights[label] for label in y_encoded])

    # 划分数据集 (贝叶斯优化只在训练集上进行，测试集留作最终验证)
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    train_weights = sample_weights[train_idx]


    # --- 定义贝叶斯优化的目标函数 ---
    def xgboost_cv(max_depth, learning_rate, n_estimators, min_child_weight, subsample, colsample_bytree):
        # 贝叶斯优化传入的是浮点数，需要转换为整数
        max_depth = int(max_depth)
        n_estimators = int(n_estimators)
        min_child_weight = int(min_child_weight)

        # 防止参数越界
        if max_depth < 1: max_depth = 1
        if n_estimators < 10: n_estimators = 10

        # 定义模型 (注意：这里不传样本权重，因为交叉验证主要看模型结构)
        # 但为了更严谨，可以在 fit 时传入，但在 bayes_opt 的包装函数中较难处理 sample_weight
        # 这里我们主要优化模型结构参数，类别权重在最终训练时加上即可
        model_temp = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=4,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1
        )

        # 使用 3折交叉验证 的准确率均值作为优化目标
        # 注意：这里暂时不加 sample_weight，为了加快优化速度。
        # 严谨的做法是在 model_temp.fit 时传入，但这需要自定义 sklearn 的 cross_val_score 逻辑
        # 对于比赛而言，结构参数的优化通常收益最大。
        score = cross_val_score(model_temp, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1).mean()

        return score


    # 定义参数搜索空间
    pbounds = {
        'max_depth': (3, 9),  # 树的最大深度
        'learning_rate': (0.01, 0.3),  # 学习率
        'n_estimators': (50, 300),  # 树的数量
        'min_child_weight': (1, 10),  # 子节点最小权重和
        'subsample': (0.7, 1.0),  # 样本采样率
        'colsample_bytree': (0.7, 1.0)  # 特征采样率
    }

    print("\n🔭 正在启动贝叶斯优化 (Bayesian Optimization)...")
    print("这可能需要几分钟到十几分钟，请耐心等待...")

    optimizer = BayesianOptimization(
        f=xgboost_cv,
        pbounds=pbounds,
        random_state=42,
        verbose=0
    )

    # 初始化 5 个点，然后迭代 15 次
    optimizer.maximize(init_points=5, n_iter=15)

    # 获取最优参数
    best_params = optimizer.max['params']
    print(f"\n🏆 贝叶斯优化找到的最优参数:")
    print(best_params)

    # ==========================================
    # 4. 使用最优参数重新训练最终模型
    # ==========================================
    print("\n🚀 正在使用最优参数训练最终模型...")

    final_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        max_depth=int(best_params['max_depth']),
        learning_rate=best_params['learning_rate'],
        n_estimators=int(best_params['n_estimators']),
        min_child_weight=int(best_params['min_child_weight']),
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False,
        verbosity=0,
        n_jobs=-1
    )

    # 这里加上样本权重进行最终训练
    final_model.fit(X_train, y_train, sample_weight=train_weights)

    # 预测
    y_pred = final_model.predict(X_test)

    print("\n✅ 训练完成！分类报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 保存模型
    model = final_model
    joblib.dump(model, 'ml_classifier_bayes_optimized.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("💾 模型已保存")

    # ==========================================
    # 5. 可视化分析
    # ==========================================
    print("\n📊 正在生成可视化图表...")
    target_names = le.classes_

    # --- 图1：混淆矩阵 ---
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('混淆矩阵 (贝叶斯优化版)', fontsize=16)
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.tight_layout()
    plt.savefig('1_混淆矩阵_贝叶斯.png', dpi=300)
    plt.show()

    # --- 图2：特征重要性 ---
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10, height=0.8, importance_type='gain')
    plt.title('特征重要性 (基于Gain)', fontsize=16)
    plt.xlabel('重要性得分', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    plt.tight_layout()
    plt.savefig('2_特征重要性_贝叶斯.png', dpi=300)
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
    plt.savefig('3_性能评估雷达图_贝叶斯.png', dpi=300)
    plt.show()

else:
    print("⚠️ 警告：没有数据留给机器学习训练。")

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
            method = "机器学习(贝叶斯优化)"
        results.append((row['Final_Label'], final_pred, method))

    print(f"{'真实标签':<20} | {'预测结果':<20} | {'判定方式'}")
    print("-" * 60)
    for real, pred, method in results:
        status = "✅" if real == pred else "❌"
        print(f"{status} {real:<18} | {pred:<18} | {method}")

print("\n🎉 流程结束！请查看生成的图片文件。")