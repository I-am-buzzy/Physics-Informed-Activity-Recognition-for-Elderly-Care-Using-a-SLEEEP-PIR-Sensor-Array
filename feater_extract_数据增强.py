import pandas as pd
import numpy as np
import os


def generate_final_robust_features():
    # --- 1. 配置 ---
    INPUT_FILE = '最终合并数据集.xlsx'
    OUTPUT_FILE = 'Dataset_Final_Robust_Features.xlsx'

    print("=" * 70)
    print("🚀 启动最终鲁棒特征工程：能量差 + 稳定性")
    print("=" * 70)

    # --- 2. 数据加载 ---
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到文件 {INPUT_FILE}")
        return

    try:
        df_raw = pd.read_excel(INPUT_FILE)
        print(f"✅ 成功加载原始数据: {df_raw.shape[0]} 行")
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    if 'Final_Label' not in df_raw.columns:
        print("❌ 错误: 找不到 'Final_Label' 列")
        return

    # --- 3. 准备数据 ---
    sensor_cols = [col for col in df_raw.columns if col != 'Final_Label']
    data = df_raw[sensor_cols].values

    # --- 4. 核心特征计算 ---
    print("⚡ 正在计算特征...")

    # A. 基础物理特征
    rms_vals = np.sqrt(np.mean(data ** 2, axis=1))
    max_vals = np.max(data, axis=1)
    min_vals = np.min(data, axis=1)
    p2p_vals = max_vals - min_vals

    # B. 区分 Class_0 vs Class_1 的关键特征
    # 1. 能量差值 (Energy_Diff)
    # 我们发现 Class_1 的能量比 Class_0 低。
    # 计算每一行能量与“全局平均能量”的差值。
    # 如果差值为负，说明比平均水平更“死寂”，可能是静止。
    global_mean_rms = np.mean(rms_vals)
    energy_diff = rms_vals - global_mean_rms

    # 2. 最大值稳定性 (Max_Stability)
    # 计算当前行的 Max 与上一行 Max 的差值绝对值。
    # 静止的人，Max 值应该很稳定（变化小）。
    # 空置时，噪声可能导致 Max 跳动。
    max_diff = np.abs(np.diff(max_vals, prepend=max_vals[0]))

    # 3. 偏度 (Skewness) - 再次尝试
    # 虽然之前说偏度差不多，但在特定能量区间内，偏度可能不同。
    # 这里我们只计算简单的偏度
    mean_vals = np.mean(data, axis=1, keepdims=True)
    std_vals = np.std(data, axis=1, keepdims=True)
    std_vals[std_vals == 0] = 1
    skewness = np.mean(((data - mean_vals) / std_vals) ** 3, axis=1)

    # --- 5. 组装结果 ---
    labels = df_raw['Final_Label'].copy()

    features_df = pd.DataFrame({
        'Label': labels,
        'RMS_Energy': rms_vals,
        'Peak_to_Peak': p2p_vals,
        'Max_Sensor': max_vals,
        # 新增特征
        'Energy_Diff': energy_diff,  # 负值越大，越像静止
        'Max_Stability': max_diff,  # 值越小，越像静止
        'Skewness': skewness
    })

    # --- 6. 保存与预览 ---
    features_df.to_excel(OUTPUT_FILE, index=False)
    print(f"\n✅ 特征提取完成！已保存至: {OUTPUT_FILE}")

    # 打印分组统计
    print("\n📊 特征预览 (按类别分组平均值):")
    print(features_df.groupby('Label').mean(numeric_only=True))

    # 重点分析
    print("\n🔍 深度对比 [空置] vs [静止]:")
    subset = features_df[features_df['Label'].isin(['Class_0_空置', 'Class_1_静止'])]
    print(subset.groupby('Label').mean(numeric_only=True)[['RMS_Energy', 'Energy_Diff', 'Max_Stability']])


if __name__ == "__main__":
    generate_final_robust_features()