import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

# Define TUM colors
class TUMcolor(tuple, Enum):
    TUMblue = (0, 101 / 255, 189 / 255)
    TUMyellow = (254 / 255, 215 / 255, 2 / 255)

# File paths
files = {
    "qwen": [
        "./data/batch_labelling_results_qwen3-0.6b_latest_20250802_135811.csv",
        "./data/batch_labelling_results_qwen3-0.6b-highD_latest_20250802_143223.csv"
    ],
    "gpt": [
        "./data/batch_labelling_results_gpt-4o_20250802_103123.csv",
        "./data/batch_labelling_results_ft_gpt-4o-2024-08-06_tum_highd_Bzt14MTi_20250802_122944.csv"
    ]
}


# Read and categorize
def read_category(files_list, category_name):
    dfs = []
    for file in files_list:
        df = pd.read_csv(file)
        df['Category'] = category_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

df_qwen = read_category(files['qwen'], 'Qwen')
df_gpt = read_category(files['gpt'], 'GPT')


# Combine and filter outliers > 30
df = pd.concat([df_qwen, df_gpt], ignore_index=True)

df = df[(df['Inference_Duration'] <= 29.9) & (df['Reach_Duration'] <= 29.9)]


for cat in ['Qwen', 'GPT']:
    cat_df = df[df['Category'] == cat]
    inf_avg = cat_df['Inference_Duration'].mean()
    inf_std = cat_df['Inference_Duration'].std()
    reach_avg = cat_df['Reach_Duration'].mean()
    reach_std = cat_df['Reach_Duration'].std()
    print(f"{cat} - Inference Duration: {inf_avg:.5f} ± {inf_std:.5f} s")
    print(f"{cat} - Reach Duration:     {reach_avg:.5f} ± {reach_std:.5f} s")
    print("-" * 50)

# Prepare data
data_inference = [df[df['Category'] == cat]['Inference_Duration'] for cat in ['Qwen', 'GPT']]
data_reach = [df[df['Category'] == cat]['Reach_Duration'] for cat in ['Qwen', 'GPT']]

positions = [1, 2, 3, 4]
colors = [TUMcolor.TUMblue.value, TUMcolor.TUMblue.value, TUMcolor.TUMyellow.value, TUMcolor.TUMyellow.value]

fig, ax = plt.subplots(figsize=(8, 2.5))
ax.grid(True, which='both', axis='x', linestyle='-', alpha=0.5)

# Plot without outlier dots
bp = ax.boxplot(
    [data_reach[0], data_inference[0], data_reach[1], data_inference[1]],
    positions=positions,
    patch_artist=True,
    vert=False,
    widths=0.8,
    showfliers=False  # <- this removes the dots!
)

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Y-tick labels
ax.set_yticks(positions)
ax.set_yticklabels([
    'Qwen Inf Duration',
    'Qwen Reach Duration',
    'GPT Inf  Duration',
    'GPT Reach Duration'
])
ax.set_xscale("log")
ax.set_xlabel('Duration (seconds)')
ax.set_title('Inference and Reach Duration by Model Category (outliers > 30 excluded, no OOD dots)')

plt.tight_layout()
plt.show()
