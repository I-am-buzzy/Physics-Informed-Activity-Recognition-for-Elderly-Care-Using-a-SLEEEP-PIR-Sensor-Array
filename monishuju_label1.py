import pandas as pd
import numpy as np
import os


def generate_static_data():
    # --- 1. 配置参数 ---
    INPUT_FILE = 'Label_1_标记后.xlsx'  # 请确保这个文件在当前目录下
    OUTPUT_FILE = '模拟生成_Class_1_静止_10000条.xlsx'
    TARGET_COUNT = 10000  # 目标总行数

    # PIR传感器列名 (根据你的Label 3文件推断是 PIR_1 到 PIR_55)
    SENSOR_COLS = [f'PIR_{i}' for i in range(1, 56)]

    print("=" * 60)
    print("🏭 开始生成 Class_1 静止模拟数据")
    print("=" * 60)

    # --- 2. 读取原始数据 ---
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到文件 {INPUT_FILE}")
        print("请将 'Label_1_标记后.xlsx' 放在与此代码相同的文件夹中。")
        return

    try:
        df_raw = pd.read_excel(INPUT_FILE)
        print(f"✅ 成功读取原始文件，共 {len(df_raw)} 行数据。")

        # 提取传感器列，如果列不存在则报错
        if not all(col in df_raw.columns for col in SENSOR_COLS):
            print("❌ 错误: 数据表中缺少 PIR 传感器列。")
            print(f"请确保列名为 {SENSOR_COLS[:3]}... 格式。")
            return

        df_static_raw = df_raw[SENSOR_COLS].copy()

    except Exception as e:
        print(f"❌ 读取文件时发生错误: {e}")
        return

    # --- 3. 数据分析与参数提取 ---
    # 静止数据的核心特征是：数值较低，且围绕一个中心值波动。
    # 我们计算每一列的均值和标准差，用于生成新数据。

    # 计算每列的统计值
    col_means = df_static_raw.mean()
    col_stds = df_static_raw.std()

    # 为了增加多样性，我们稍微扩大一点标准差（乘以1.2），模拟不同环境下的静止背景
    noise_scale = col_stds * 1.2

    print(f"📊 正在分析数据特征... 共 {len(SENSOR_COLS)} 个传感器通道。")

    # --- 4. 生成模拟数据 ---
    synthetic_data = []
    current_count = len(df_static_raw)
    needed_count = TARGET_COUNT - current_count

    print(f"⚡ 正在生成 {needed_count} 条新数据...")

    for _ in range(needed_count):
        # 生成一行新数据
        # np.random.normal 生成符合正态分布的随机数
        # loc=均值, scale=波动范围
        new_row = np.random.normal(loc=col_means, scale=noise_scale)

        # 静止数据不能是负数（传感器没有负值），也不能超过最大值
        # 16383 是你提供的 Label 3 数据中的最大值，我们沿用
        new_row = np.clip(new_row, 0, 16383)

        # 为了模拟真实的“死寂”状态，我们可以随机将一些极低的值强制归零
        # 模拟传感器的信噪比阈值
        new_row = np.where(new_row < 50, 0, new_row)

        synthetic_data.append(new_row)

    # --- 5. 组装最终数据集 ---
    # 1. 创建模拟数据的 DataFrame
    df_synthetic = pd.DataFrame(synthetic_data, columns=SENSOR_COLS)

    # 2. 将原始数据和模拟数据合并
    # ignore_index=True 重新生成索引
    df_final = pd.concat([df_static_raw, df_synthetic], ignore_index=True)

    # 3. 添加标签列
    df_final['Final_Label'] = 'Class_1_静止'

    # 4. 打乱顺序 (防止前800条全是原始数据，后面全是模拟数据)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- 6. 保存文件 ---
    try:
        df_final.to_excel(OUTPUT_FILE, index=False)
        print("\n" + "=" * 60)
        print("✅ 生成成功！")
        print("=" * 60)
        print(f"📄 文件名: {OUTPUT_FILE}")
        print(f"📊 总行数: {len(df_final)} (原始: {current_count} + 模拟: {needed_count})")
        print(f"🔍 标签: {df_final['Final_Label'].iloc[0]}")

        # 打印前5行预览
        print("\n📋 数据预览 (前5行):")
        print(df_final.head())

    except Exception as e:
        print(f"❌ 保存文件时发生错误: {e}")


if __name__ == "__main__":
    generate_static_data()