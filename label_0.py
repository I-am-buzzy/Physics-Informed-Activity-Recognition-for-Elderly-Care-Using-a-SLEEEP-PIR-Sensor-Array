import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.dates import DateFormatter, HourLocator

# --------------------------
# 1. 解决中文显示（必须替换为你的字体路径！）
# --------------------------
# Windows默认SimHei路径：C:/Windows/Fonts/simhei.ttf
# macOS默认PingFang SC路径：/Library/Fonts/PingFang SC.ttc
# Linux需先安装中文字体，如wqy-microhei：/usr/share/fonts/truetype/wqy/wqy-microhei.ttc
font_path = "C:/Windows/Fonts/simhei.ttf"  # 请根据系统修改
font = FontProperties(fname=font_path, size=10)
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示
plt.style.use('seaborn-v0_8-ticks')


# --------------------------
# 2. 读取并预处理数据
# --------------------------
# 读取Excel文件（替换为你的文件路径）
df = pd.read_excel("拆分结果_Label_3_运动.xlsx", sheet_name="Sheet1")

# 处理时间列：将字符串转换为datetime格式（假设Time列格式为"HH:MM:SS"）
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')

# 提取核心数据列
time_data = df['Time']  # 时间序列
temp_data = df['Temperature_F']  # 温度数据
pir_columns = [f"PIR_{i}" for i in range(1, 56)]  # PIR_1到PIR_55列
pir_data = df[pir_columns]  # PIR传感器数据


# --------------------------
# 3. 可视化1：温度随时间的波动趋势
# --------------------------
plt.figure(figsize=(12, 4))
plt.plot(time_data, temp_data, color='#FF6B6B', linewidth=1.5, alpha=0.8)
plt.title("温度随时间变化趋势", fontproperties=font, fontweight='bold', pad=20)
plt.xlabel("时间", fontproperties=font)
plt.ylabel("温度 (°F)", fontproperties=font)
plt.grid(True, alpha=0.3)
# 格式化x轴时间显示（仅显示小时:分钟）
plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(HourLocator(interval=1))  # 每小时标记一次
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("1_温度时间趋势.png", dpi=300, bbox_inches='tight')
plt.close()


# --------------------------
# 4. 可视化2：PIR传感器空间分布（平均值+标准差）
# --------------------------
# 计算PIR传感器的统计量
pir_stats = pd.DataFrame({
    "平均值": pir_data.mean(),
    "标准差": pir_data.std()
}).sort_index()  # 按传感器编号排序

plt.figure(figsize=(12, 5))
x = range(1, 56)  # 传感器编号1-55
# 绘制平均值折线
plt.plot(x, pir_stats["平均值"], color='#4ECDC4', linewidth=2, marker='o', markersize=4, label="平均值")
# 绘制标准差填充区域（表示波动范围）
plt.fill_between(x,
                 pir_stats["平均值"] - pir_stats["标准差"],
                 pir_stats["平均值"] + pir_stats["标准差"],
                 color='#4ECDC4', alpha=0.2, label="±1标准差")
plt.title("PIR传感器数值空间分布（平均值±标准差）", fontproperties=font, fontweight='bold', pad=20)
plt.xlabel("传感器编号", fontproperties=font)
plt.ylabel("PIR数值", fontproperties=font)
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 56, 5))  # 每5个传感器标记一次
plt.legend(prop=font)
plt.tight_layout()
plt.savefig("2_PIR空间分布.png", dpi=300, bbox_inches='tight')
plt.close()


# --------------------------
# 5. 可视化3：PIR传感器时间序列热力图
# --------------------------
# 取前200行数据（避免图表过于密集，可根据需求调整）
heatmap_data = pir_data.iloc[:200, :].T  # 转置：传感器为行，时间为列
time_labels = time_data.iloc[:200].dt.strftime('%H:%M')  # 时间标签格式化为"小时:分钟"

plt.figure(figsize=(14, 8))
norm = plt.Normalize(vmin=pir_data.min().min(), vmax=pir_data.max().max())  # 数值归一化
im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto', norm=norm)

# 设置标题和标签
plt.title("PIR传感器时间序列热力图（前200行）", fontproperties=font, fontweight='bold', pad=20)
plt.xlabel("时间", fontproperties=font)
plt.ylabel("传感器编号", fontproperties=font)
# 设置x轴时间标签（每20行显示一次，避免拥挤）
plt.xticks(range(0, 200, 20), time_labels[::20], rotation=45, fontproperties=font)
plt.yticks(range(0, 55, 5), range(1, 56, 5), fontproperties=font)  # y轴每5个传感器标记一次

# 添加颜色条
cbar = plt.colorbar(im, shrink=0.8, pad=0.05)
cbar.set_label("PIR数值", fontproperties=font)
plt.tight_layout()
plt.savefig("3_PIR时间热力图.png", dpi=300, bbox_inches='tight')
plt.close()


# --------------------------
# 6. 可视化4：温度与PIR数据的相关性（可选）
# --------------------------
# 计算温度与每个PIR传感器的相关系数
corr_temp_pir = [temp_data.corr(pir_data[col]) for col in pir_columns]

plt.figure(figsize=(12, 4))
plt.bar(x, corr_temp_pir, color='#FFA500', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)  # 零相关线
plt.title("温度与各PIR传感器的相关性", fontproperties=font, fontweight='bold', pad=20)
plt.xlabel("传感器编号", fontproperties=font)
plt.ylabel("相关系数", fontproperties=font)
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(range(0, 56, 5))
plt.tight_layout()
plt.savefig("4_温度PIR相关性.png", dpi=300, bbox_inches='tight')
plt.close()


# --------------------------
# 输出关键统计结果（可选）
# --------------------------
print("=== 温度统计 ===")
print(f"平均温度：{temp_data.mean():.2f}°F，温度范围：{temp_data.min()}~{temp_data.max()}°F")
print("\n=== PIR传感器统计（前5个）===")
print(pir_stats.head().round(2))