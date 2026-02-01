import pandas as pd
import os

# 数据目录
data_dir = "./data"

# 布尔列
bool_cols = ['Safe_Top1', 'Safe_TopK', 'MONA_safe', 'Match_Top1', 'Match_TopK']

# 存储所有文件结果
longitudinal_stats = []
lateral_stats = []

# 遍历所有 CSV 文件
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            try:
                df = pd.read_csv(file_path)
                # Longitudinal 分组计算比例
                longitudinal = df.groupby('Trajectory_Longitudinal')[bool_cols].mean() * 100
                longitudinal_stats.append(longitudinal)

                # Lateral 分组计算比例
                lateral = df.groupby('Trajectory_Lateral')[bool_cols].mean() * 100
                lateral_stats.append(lateral)
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

# 合并所有文件，按动作类型求平均
longitudinal_avg = pd.concat(longitudinal_stats).groupby(level=0).mean()
lateral_avg = pd.concat(lateral_stats).groupby(level=0).mean()

# 四舍五入保留两位小数
longitudinal_avg = longitudinal_avg.round(2)
lateral_avg = lateral_avg.round(2)

# 输出
print("=== Average percentages for Trajectory_Longitudinal ===")
print(longitudinal_avg)
print("\n=== Average percentages for Trajectory_Lateral ===")
print(lateral_avg)

# 可选：保存到 CSV
longitudinal_avg.to_csv("longitudinal_avg.csv")
lateral_avg.to_csv("lateral_avg.csv")
