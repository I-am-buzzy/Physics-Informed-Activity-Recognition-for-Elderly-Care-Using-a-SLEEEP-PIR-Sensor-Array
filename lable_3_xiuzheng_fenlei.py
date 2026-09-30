import pandas as pd
import numpy as np


def physics_based_fall_detection():
    print("正在加载数据...")
    try:
        df = pd.read_excel('Label_3_最终修正版_增加大幅度运动.xlsx')
    except FileNotFoundError:
        print("错误：找不到文件")
        return

    pir_cols = [f'PIR_{i}' for i in range(1, 56)]
    data = df[pir_cols].values.astype(float)
    n_rows = len(data)

    results = []
    noise_floor = 3000

    print("正在进行基于物理特性的跌倒检测...")

    for i in range(n_rows):
        row = data[i]
        valid_signals = row[row > noise_floor]

        if len(valid_signals) == 0:
            results.append("✅ 正常静止")
            continue

        max_val = np.max(valid_signals)
        width = len(valid_signals)  # 覆盖的传感器数量
        energy_sum = np.sum(valid_signals)

        # --- 物理逻辑修正 ---
        # 1. 如果数值极高，这在物理上只能是"大幅度运动"，绝不是跌倒。
        #    因为跌倒后人体变远，数值会降低，不会维持在 15000+。
        if max_val > 15000:
            results.append("🏃 大幅度运动 (剧烈动作/离得太近)")

        # 2. 跌倒的特殊判定：需要结合时序。
        #    如果当前行数值不是极高，但是覆盖宽度非常大（比如 > 40 个传感器都有微弱信号），
        #    这才可能是平躺（平躺覆盖广，但信号弱）。
        elif width > 40 and max_val < 8000:
            results.append("⚠️ 跌倒 (平躺覆盖广但信号弱)")

        # 3. 普通运动
        else:
            results.append("✅ 正常运动 (走动)")

    # --- 时序修正 (检查是否有 "高-低" 跌倒模式) ---
    # 遍历结果，寻找 "大幅度运动" -> "跌倒" 的瞬间切换
    for i in range(1, n_rows):
        # 如果上一秒是剧烈运动（摔倒瞬间），这一秒变成了平躺（宽幅低值）
        if (results[i - 1] == "🏃 大幅度运动 (剧烈动作/离得太近)" and
                results[i] == "⚠️ 跌倒 (平躺覆盖广但信号弱)"):
            # 确认这是一个跌倒事件
            results[i] = "🚨 确认跌倒 (摔倒瞬间后平躺)"
            # 如果前一行是剧烈运动，也标记为跌倒过程的一部分
            if i > 0:
                results[i - 1] = "⚡ 跌倒瞬间 (爆发)"

    df['物理特性分类'] = results
    df.to_excel('Label_3_增强物理特性修正分类结果.xlsx', index=False)

    print("\n--- 统计结果 ---")
    print(pd.Series(results).value_counts())
    print("\n完成！")


physics_based_fall_detection()