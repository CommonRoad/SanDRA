import pandas as pd

# 读取 CSV
csv_file = "./data/rule_prompt/batch_labelling_results_gpt-4o_20250912_102453-rule_True.csv"
# csv_file = './data/commands/batch_labelling_results_gpt-4o_20250724_104647.csv'
df = pd.read_csv(csv_file)

# 你要统计的列
bool_cols = ['Safe_Top1', 'Safe_TopK', 'MONA_safe', 'Match_Top1', 'Match_TopK']

# 统计 Trajectory_Longitudinal 各类别比例
longitudinal_stats = df.groupby('Trajectory_Longitudinal')[bool_cols].mean() * 100
print("=== Trajectory_Longitudinal percentages ===")
print(longitudinal_stats.round(2))

# 统计 Trajectory_Lateral 各类别比例
lateral_stats = df.groupby('Trajectory_Lateral')[bool_cols].mean() * 100
print("\n=== Trajectory_Lateral percentages ===")
print(lateral_stats.round(2))
