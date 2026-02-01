import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
DATA_DIR = "./data/horizon"

LONG_COL = "qwen3-0.6b:latest_Longitudinal_1"
LAT_COL  = "qwen3-0.6b:latest_Lateral_1"

# =========================
# Load CSV files
# =========================
csv_files = sorted([
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith(".csv")
])

if len(csv_files) != 4:
    raise ValueError(f"Expected 4 CSV files, got {len(csv_files)}")

print("Loaded CSV files:")
for f in csv_files:
    print(" -", os.path.basename(f))


# =========================
# Helper: build ratio table
# =========================
def build_ratio_df(column_name):
    records = []

    for path in csv_files:
        df = pd.read_csv(path)
        fname = os.path.basename(path)

        if column_name not in df.columns:
            raise KeyError(f"{column_name} not found in {fname}")

        # normalized value counts → proportions
        vc = df[column_name].value_counts(normalize=True)

        for action, ratio in vc.items():
            records.append({
                "file": fname,
                "action": action,
                "ratio": ratio
            })

    ratio_df = pd.DataFrame(records)

    # pivot to: rows = file, cols = action
    pivot_df = ratio_df.pivot(
        index="file",
        columns="action",
        values="ratio"
    ).fillna(0)

    return pivot_df


# =========================
# Longitudinal actions
# =========================
long_df = build_ratio_df(LONG_COL)

plt.figure(figsize=(9, 4))
long_df.plot(
    kind="bar",
    stacked=True,
    ax=plt.gca()
)

plt.ylabel("Proportion")
plt.xlabel("CSV file (different horizons)")
plt.title("Longitudinal Action Distribution (Qwen-0.6B)")
plt.legend(title="Action", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.show()


# =========================
# Lateral actions
# =========================
lat_df = build_ratio_df(LAT_COL)

plt.figure(figsize=(9, 4))
lat_df.plot(
    kind="bar",
    stacked=True,
    ax=plt.gca()
)

plt.ylabel("Proportion")
plt.xlabel("CSV file (different horizons)")
plt.title("Lateral Action Distribution (Qwen-0.6B)")
plt.legend(title="Action", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.show()
