import pandas as pd
import numpy as np
import os


def merge_and_process_data():
    print("=" * 60)
    print("🚀 启动数据合并流水线 (四类独立 + 时间排序)")
    print("=" * 60)

    # --- 1. 定义文件路径 ---
    file_0 = 'Label_0_标记后.xlsx'
    file_1 = 'Label_1_标记后.xlsx'
    file_3 = 'Label_3_物理特性修正分类结果.xlsx'
    output_file = '最终合并数据集_原始数据.xlsx'

    # 检查文件是否存在
    missing_files = [f for f in [file_0, file_1, file_3] if not os.path.exists(f)]
    if missing_files:
        print(f"❌ 错误：找不到文件 {missing_files}，请确保文件在当前目录下。")
        return

    # --- 2. 读取数据 ---
    print("📂 正在读取 Excel 文件...")
    df0 = pd.read_excel(file_0)
    df1 = pd.read_excel(file_1)
    df3 = pd.read_excel(file_3)

    # --- 3. 时间列处理与类型转换 ---
    time_col = 'Time'
    # 统一转换时间格式
    for df in [df0, df1, df3]:
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        else:
            # 如果没有时间列，填充当前时间以便排序（通常Label 0/1应该没有具体时间）
            df[time_col] = pd.Timestamp('now')

    print(f"   - Label 0 (空置): {len(df0)} 行")
    print(f"   - Label 1 (静止): {len(df1)} 行")
    print(f"   - Label 3 (复杂动作): {len(df3)} 行")

    # --- 4. 统一标签映射 (四类独立) ---
    print("\n🏷️ 正在统一标签分类...")

    # 定义映射字典 (关键：正常运动、大幅度运动、跌倒 互不合并)
    map_3 = {
        '✅ 正常静止': 'Class_1_静止',
        '✅ 正常运动 (走动)': 'Class_2_正常运动',  # 独立类别
        '🏃 大幅度运动 (剧烈动作/离得太近)': 'Class_3_大幅度运动',  # 独立类别
        '⚠️ 跌倒 (平躺覆盖广但信号弱)': 'Class_4_跌倒',  # 独立类别
        '🚨 确认跌倒 (摔倒瞬间后平躺)': 'Class_4_跌倒',  # 独立类别
        '⚡ 跌倒瞬间 (爆发)': 'Class_4_跌倒'  # 独立类别
    }

    # 应用映射
    df0['Final_Label'] = 'Class_0_空置'
    df1['Final_Label'] = 'Class_1_静止'
    df3['Final_Label'] = df3['物理特性分类'].map(map_3)

    # --- 5. 合并数据 ---
    print("\n🔗 正在合并数据...")
    pir_cols = [f'PIR_{i}' for i in range(1, 56)]

    # 确保列对齐
    for df in [df0, df1, df3]:
        for col in pir_cols:
            if col not in df.columns:
                df[col] = 0

    data_0 = df0[pir_cols + [time_col, 'Final_Label']].copy()
    data_1 = df1[pir_cols + [time_col, 'Final_Label']].copy()
    data_3 = df3[pir_cols + [time_col, 'Final_Label']].copy()

    df_all = pd.concat([data_0, data_1, data_3], ignore_index=True)
    print(f"   - 合并后总行数: {len(df_all)}")

    # --- 6. 特征工程 (修复版) ---
    print("\n⚙️ 正在进行特征提取...")
    features_list = []
    noise_floor = 3000

    for index, row in df_all.iterrows():
        sensors = row[pir_cols].values.astype(float)
        valid_mask = sensors > noise_floor
        valid_sensors = sensors[valid_mask]
        valid_indices = np.where(valid_mask)[0]  # 修正索引

        features = {}

        # 强度特征
        features['max_val'] = valid_sensors.max() if len(valid_sensors) > 0 else 0
        features['mean_val'] = valid_sensors.mean() if len(valid_sensors) > 0 else 0
        features['sum_val'] = valid_sensors.sum()
        features['std_val'] = sensors.std()

        # 空间特征
        features['width'] = len(valid_sensors)

        # 重心计算 (修正权重索引)
        if len(valid_indices) > 0:
            features['center'] = np.average(valid_indices, weights=valid_sensors)  # 修正行
            features['span'] = valid_indices[-1] - valid_indices[0]
        else:
            features['center'] = 0
            features['span'] = 0

        features_list.append(features)

    # --- 7. 组装与排序 ---
    print("\n📅 正在按时间顺序排序...")
    df_features = pd.DataFrame(features_list)

    # 合并
    df_final = pd.concat([df_all[pir_cols + [time_col, 'Final_Label']], df_features], axis=1)

    # 关键：按时间排序 (去除无效时间行)
    df_final = df_final.dropna(subset=[time_col])
    df_final = df_final.sort_values(by=time_col).reset_index(drop=True)

    # --- 8. 保存 ---
    print(f"\n💾 正在保存最终数据集...")
    df_final.to_excel(output_file, index=False)

    print("\n" + "=" * 60)
    print("✅ 处理完成！")
    print("=" * 60)
    print(f"📊 最终数据分布统计:")
    print(df_final['Final_Label'].value_counts())
    print(f"📄 文件已保存为: {output_file}")


if __name__ == "__main__":
    merge_and_process_data()