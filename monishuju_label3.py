import pandas as pd
import numpy as np
import os


def generate_final_aligned_data():
    # --- 1. 配置参数 ---
    REAL_DATA_FILE = 'Label_3_运动_清洗后.xlsx'
    OUTPUT_FILE = 'Label_3_最终修正版_增加大幅度运动.xlsx'

    TARGET_TOTAL = 12000  # 目标总行数
    SENSOR_COUNT = 55

    # --- 2. 读取真实数据 ---
    try:
        df_real = pd.read_excel(REAL_DATA_FILE)
        real_row_count = len(df_real)
        print(f"✅ 读取真实数据: {real_row_count} 行")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到真实数据文件。")
        df_real = pd.DataFrame()
        real_row_count = 0

    # --- 3. 计算模拟数量 (关键修改点) ---
    remaining_slots = TARGET_TOTAL - real_row_count

    # 跌倒：极低概率，仅 2 条 (保持不变)
    synthetic_fall = 2

    # 大幅度运动：增加到 500 条 (之前的 5 倍，模拟更多剧烈动作)
    synthetic_large = 500

    # 正常运动：剩下的所有名额 (会自动减少)
    synthetic_normal = remaining_slots - synthetic_fall - synthetic_large

    print(f"📊 生成计划:")
    print(f"   - 真实数据: {real_row_count}")
    print(f"   - 模拟正常运动 (高数值): {synthetic_normal} (数值 8500-14000)")
    print(f"   - 模拟跌倒 (低数值): {synthetic_fall} (数值 0-2000)")
    print(f"   - 模拟大幅度 (极高数值): {synthetic_large} (数值 15000-16383)")
    print("-" * 50)

    # --- 4. 生成数据 (关键修正：数值区间互斥) ---
    data_list = []

    # 1. 生成正常运动：数值高，避开跌倒判定
    print("⚡ 生成正常运动数据 (高数值)...")
    for i in range(synthetic_normal):
        row = {'Time': f'Norm_H_{i}', 'Label': 3}
        # 生成高数值数据 (8500-14000)，确保 > 8000
        for j in range(1, SENSOR_COUNT + 1):
            row[f'PIR_{j}'] = np.random.randint(8500, 14000)
        data_list.append(row)

    # 2. 生成大幅度运动：数值极高
    print("⚡ 生成大幅度运动数据...")
    for i in range(synthetic_large):
        row = {'Time': f'Large_{i}', 'Label': 3}
        # 生成极高数值数据 (15000-16383)
        for j in range(1, SENSOR_COUNT + 1):
            row[f'PIR_{j}'] = np.random.randint(15000, 16383)
        data_list.append(row)

    # 3. 生成跌倒：数值极低
    print("⚡ 生成跌倒数据 (低数值)...")
    for i in range(synthetic_fall):
        row = {'Time': f'Fall_L_{i}', 'Label': 3}
        # 生成极低数值数据 (0-2000)
        for j in range(1, SENSOR_COUNT + 1):
            row[f'PIR_{j}'] = np.random.randint(0, 2000)
        data_list.append(row)

    # --- 5. 合并与保存 ---
    df_syn = pd.DataFrame(data_list)

    # 处理真实数据列名
    if not df_real.empty:
        cols = ['Time', 'Label'] + [f'PIR_{i}' for i in range(1, SENSOR_COUNT + 1)]
        df_real = df_real.reindex(columns=cols, fill_value=0)

    final_df = pd.concat([df_real, df_syn], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    final_df.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ 数据生成完成！文件名: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_final_aligned_data()