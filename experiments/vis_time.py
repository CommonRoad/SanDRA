import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

# =========================
# TUM colors
# =========================
class TUMcolor(tuple, Enum):
    TUMblue = (0, 101 / 255, 189 / 255)
    TUMyellow = (254 / 255, 215 / 255, 2 / 255)


# =========================
# File paths
# =========================
files = {
    "qwen": [
        "./data/LLMs/batch_labelling_results_qwen3-0.6b_latest_20250802_135811.csv",
        "./data/LLMs/batch_labelling_results_qwen3-0.6b-highD_latest_20250802_143223.csv"
    ],
    "gpt": [
        "./data/LLMs/batch_labelling_results_gpt-4o_20250802_103123.csv",
        "./data/LLMs/batch_labelling_results_ft_gpt-4o-2024-08-06_tum_highd_Bzt14MTi_20250802_122944.csv"
    ],
    "no_LLM": [
        "./data/LLMs/no-early-stopping.csv"
    ]
}


# =========================
# Helper: read & tag
# =========================
def read_category(file_list, category_name):
    dfs = []
    for f in file_list:
        df = pd.read_csv(f)
        df["Category"] = category_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# =========================
# Load data
# =========================
df_qwen = read_category(files["qwen"], "Qwen")
df_gpt = read_category(files["gpt"], "GPT")
df_no_llm = read_category(files["no_LLM"], "No-LLM")

# ---- IMPORTANT ----
# If your no_LLM csv uses a different column name, change THIS line only:
# e.g. df_no_llm["Reach_Duration"] = df_no_llm["Duration"]
# -------------------
df_no_llm["Inference_Duration"] = pd.NA  # no inference for no-LLM

# =========================
# Combine
# =========================
df = pd.concat([df_qwen, df_gpt, df_no_llm], ignore_index=True)

# =========================
# Outlier filtering
# - LLM runs: filter >30s
# - no_LLM: keep ALL (important!)
# =========================
df = df[
    (df["Category"] == "No-LLM") |
    (
        ((df["Inference_Duration"].isna()) | (df["Inference_Duration"] <= 29.9)) &
        (df["Reach_Duration"] <= 29.9)
    )
]

# =========================
# Sanity check (DO NOT REMOVE)
# =========================
print("Counts per category:")
print(df.groupby("Category").size())
print("\nno_LLM Reach stats:")
print(df[df["Category"] == "No-LLM"]["Reach_Duration"].describe())


# =========================
# Prepare boxplot data (safe)
# =========================
def safe_series(series):
    series = series.dropna()
    return series if len(series) > 0 else None


data = [
    safe_series(df[df["Category"] == "Qwen"]["Reach_Duration"]),
    safe_series(df[df["Category"] == "Qwen"]["Inference_Duration"]),
    safe_series(df[df["Category"] == "GPT"]["Reach_Duration"]),
    safe_series(df[df["Category"] == "GPT"]["Inference_Duration"]),
    safe_series(df[df["Category"] == "No-LLM"]["verification-time"]),
]

labels = [
    "Qwen Reach Duration",
    "Qwen Inference Duration",
    "GPT Reach Duration",
    "GPT Inference Duration",
    "No-LLM Reach Duration",
]

colors = [
    TUMcolor.TUMblue.value,
    TUMcolor.TUMblue.value,
    TUMcolor.TUMyellow.value,
    TUMcolor.TUMyellow.value,
    (0.6, 0.6, 0.6),  # gray for no-LLM
]

# Remove None entries (extra safety)
data_final, labels_final, colors_final = [], [], []
for d, l, c in zip(data, labels, colors):
    if d is not None:
        data_final.append(d)
        labels_final.append(l)
        colors_final.append(c)

positions = list(range(1, len(data_final) + 1))


# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(8, 3))
ax.grid(True, which="both", axis="x", linestyle="-", alpha=0.4)

bp = ax.boxplot(
    data_final,
    positions=positions,
    vert=False,
    patch_artist=True,
    widths=0.75,
    showfliers=False
)

for patch, color in zip(bp["boxes"], colors_final):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_yticks(positions)
ax.set_yticklabels(labels_final)

ax.set_xscale("log")
ax.set_xlabel("Duration (seconds)")
ax.set_title(
    "Inference and Reachability Runtime Comparison\n"
    "(no-LLM = exhaustive enumeration of all 12 actions)"
)

plt.tight_layout()
plt.show()
