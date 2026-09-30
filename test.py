import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import fft
import warnings

warnings.filterwarnings('ignore')


# -----------------------------
# 1. 数据加载 (修正版)
# -----------------------------
def load_and_explore_data(file_path):
    """
    修正重点：处理表头和类型转换
    """
    # 尝试1：假设第一行是表头 (大多数Kaggle/UCI数据集都是这样)
    # 如果读取后发现列名是 '2015-02-18' 这种数字，说明假设错了，需要改用 header=None
    df = pd.read_csv(file_path)

    # 检查第一列是否是日期格式，如果是，说明我们读对了表头
    # 如果第一列看起来像数字或者乱码，可能需要调整
    print("读取到的列名:", df.columns.tolist())

    # 如果列名不对（比如列名变成了具体的数据），请取消下面这行的注释并重新运行
    # df = pd.read_csv(file_path, header=None, names=['Date', 'Time', 'Label', 'Temperature_F'] + [f'PIR_{i}' for i in range(1, 56)])

    # 强制转换类型：防止温度列被读成字符串
    # errors='coerce' 会把无法转换的字符变成 NaN，避免报错
    if 'Temperature_F' in df.columns:
        df['Temperature_F'] = pd.to_numeric(df['Temperature_F'], errors='coerce')

    # 确保 PIR 列也是数字
    pir_cols = [col for col in df.columns if 'PIR' in str(col)]
    for col in pir_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print("数据形状:", df.shape)
    print("温度列前5行:", df['Temperature_F'].head())

    return df


# -----------------------------
# 2. 数据清洗 (保持不变，但增加了安全检查)
# -----------------------------
def clean_data(df):
    """
    1. 处理温度单位
    2. 处理PIR传感器的饱和值 (16383)
    """
    # 2.1 温度转换
    # 检查是否有非空数据再计算，防止全空报错
    if df['Temperature_F'].notna().any():
        df['Temperature_C'] = (df['Temperature_F'] - 32) * 5 / 9
    else:
        print("警告：温度数据全为空！")
        df['Temperature_C'] = np.nan

    # 2.2 处理 PIR 异常值 (16383)
    # 找到所有 PIR 列
    pir_columns = [col for col in df.columns if 'PIR' in str(col)]

    # 替换异常值
    for col in pir_columns:
        df[col] = df[col].replace(16383, np.nan)

    # 使用线性插值填充
    # axis=1 表示按行插值
    df[pir_columns] = df[pir_columns].apply(lambda row: row.interpolate(method='linear', limit_direction='both'),
                                            axis=1)

    # 如果插值后还有空值（比如整行都是16383），则删除该行
    df = df.dropna(subset=pir_columns)

    return df


# -----------------------------
# 3. 特征工程 (保持不变)
# -----------------------------
def extract_features(row):
    """
    提取统计特征
    """
    # 动态获取 PIR 列，防止列名不匹配
    pir_cols = [col for col in row.index if 'PIR' in str(col)]
    # 确保按顺序排列 (PIR_1, PIR_2...)
    pir_cols.sort(key=lambda x: int(x.split('_')[1]))

    signal = row[pir_cols].values.astype(float)

    # 基础统计特征
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    skew_val = stats.skew(signal)
    kurt_val = stats.kurtosis(signal)
    rms = np.sqrt(np.mean(signal ** 2))

    # 频域特征
    fft_vals = fft(signal)
    fft_magnitude = np.abs(fft_vals[:len(fft_vals) // 2])
    dominant_freq_energy = np.sum(np.sort(fft_magnitude)[-5:])

    # 过零率
    zcr = ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)

    # 温度残差
    temp = row['Temperature_C'] if 'Temperature_C' in row else 0

    features = {
        'feat_mean': mean_val,
        'feat_std': std_val,
        'feat_max': max_val,
        'feat_min': min_val,
        'feat_skew': skew_val,
        'feat_kurt': kurt_val,
        'feat_rms': rms,
        'feat_dominant_freq': dominant_freq_energy,
        'feat_zcr': zcr,
        'feat_temp_residual': mean_val - (0.1 * temp),
        'original_label': row['Label']
    }

    return pd.Series(features)


# -----------------------------
# 4. 主程序
# -----------------------------
def main():
    file_path = 'dataset.csv'  # 确保文件名和路径正确

    # 步骤1: 加载 (已修复)
    df_raw = load_and_explore_data(file_path)

    # 步骤2: 清洗
    df_clean = clean_data(df_raw)

    # 步骤3: 特征提取
    print("\n正在提取特征...")
    df_features = df_clean.apply(extract_features, axis=1)

    # 步骤4: 保存
    output_path = 'processed_features.csv'
    df_features.to_csv(output_path, index=False)
    print(f"\n成功！特征数据已保存至: {output_path}")
    print(f"特征矩阵形状: {df_features.shape}")


if __name__ == "__main__":
    main()