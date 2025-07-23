import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt


def plot_aggregate_verified_id_histogram(base_dir, model_name, lanes_count, vehicles_density, use_sonia=False,
                                         save_path=None):
    """
    Plots a histogram aggregating 'verified-id' values from all matching evaluation.csv files.

    Args:
        base_dir (str or Path): Directory containing the result folders.
        model_name (str): Model name used in folder naming.
        lanes_count (int): Number of lanes.
        vehicles_density (float): Traffic density.
        use_sonia (bool): Whether the folder name includes '-spot'.
        save_path (str or Path): Optional path to save the plot.
    """
    base_path = Path(base_dir)
    if use_sonia:
        pattern_str = rf"results-True-{re.escape(model_name)}-{lanes_count}-{vehicles_density}-\d+-spot"
    else:
        pattern_str = rf"results-True-{re.escape(model_name)}-{lanes_count}-{vehicles_density}-\d+"

    pattern = re.compile(pattern_str)

    # Initialize count for each verified-id class
    total_counts = pd.Series([0, 0, 0, 0], index=[0, 1, 2, 3], dtype=int)

    for folder in base_path.iterdir():
        if folder.is_dir() and pattern.fullmatch(folder.name):
            csv_file = folder / "evaluation.csv"
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    if "verified-id" in df.columns:
                        values = df["verified-id"].dropna().astype(int)
                        counts = values.value_counts().reindex([0, 1, 2, 3], fill_value=0)
                        total_counts += counts
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")

    # Plot the aggregated histogram
    plt.figure(figsize=(6, 4))
    bars = plt.bar(total_counts.index, total_counts.values, color=["green", "blue", "orange", "red"])
    plt.xticks([0, 1, 2, 3], ["0", "1", "2", "3 (fail-safe)"])
    plt.xlabel("verified-id")
    plt.ylabel("Total Count")
    plt.title(f"Aggregated verified-id Distribution ({model_name}, lanes={lanes_count}, density={vehicles_density})")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{int(height)}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def average_travelled_value(base_dir, model_name, lanes_count, vehicles_density, use_sonia=False):
    base_path = Path(base_dir)
    # Adjust regex based on use_sonia flag
    if use_sonia:
        pattern_str = rf"results-True-{re.escape(model_name)}-{lanes_count}-{vehicles_density}-\d+-spot"
    else:
        pattern_str = rf"results-True-{re.escape(model_name)}-{lanes_count}-{vehicles_density}-\d+"

    pattern = re.compile(pattern_str)

    travelled_values = []

    for folder in base_path.iterdir():
        if folder.is_dir() and pattern.fullmatch(folder.name):
            csv_file = folder / "evaluation.csv"
            if csv_file.exists():
                try:
                    # Read CSV with no headers, infer separator (change sep if needed)
                    df = pd.read_csv(csv_file, header=None)
                    # Get last row as a Series
                    last_row = df.iloc[-1]
                    val = last_row[1]  # second column in last row
                    # Extract last numeric value from last row
                    # Convert all entries to strings, filter numeric-like entries, pick last one
                    try:
                        numeric_val = float(val)
                        travelled_values.append(numeric_val)
                    except (ValueError, TypeError):
                        print(f"Warning: Non-numeric travelled value '{val}' in file {csv_file}")
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")

    if travelled_values:
        avg_travelled = sum(travelled_values) / len(travelled_values)
        return avg_travelled
    else:
        print("No travelled values found.")
        return None


def average_travelled_value_finished(base_dir, model_name, lanes_count, vehicles_density, use_sonia=False):

    base_path = Path(base_dir)
    # Adjust regex based on use_sonia flag
    if use_sonia:
        pattern_str = rf"results-True-{re.escape(model_name)}-{lanes_count}-{vehicles_density}-\d+-spot"
    else:
        pattern_str = rf"results-True-{re.escape(model_name)}-{lanes_count}-{vehicles_density}-\d+"

    pattern = re.compile(pattern_str)
    travelled_values = []

    for folder in base_path.iterdir():
        if folder.is_dir() and pattern.fullmatch(folder.name):
            csv_file = folder / "evaluation.csv"
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file, sep=None, engine='python')  # Let pandas infer delimiter
                    if "iteration-id" in df.columns:
                        if pd.to_numeric(df["iteration-id"], errors="coerce").max() == 31.:
                            val = df.iloc[-1, 1]  # second column (assumed to be the travelled value)
                            try:
                                numeric_val = float(val)
                                travelled_values.append(numeric_val)
                            except (ValueError, TypeError):
                                print(f"Warning: Non-numeric travelled value '{val}' in file {csv_file}")
                    else:
                        print(f"Warning: 'iteration-id' not found in {csv_file}")
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")

    if travelled_values:
        avg_travelled = sum(travelled_values) / len(travelled_values)
        return avg_travelled
    else:
        print("No travelled values found where iteration-id reached 31.")
        return None


def average_max_iteration_id(base_dir, model_name, lanes_count, vehicles_density, use_sonia=False):
    """
    Computes the average of the maximum 'iteration-id' values across all evaluation.csv files.

    Args:
        base_dir (str or Path): Directory containing the result folders.
        model_name (str): Model name used in folder naming.
        lanes_count (int): Number of lanes.
        vehicles_density (float): Traffic density.
        use_sonia (bool): Whether the folder name includes '-spot'.

    Returns:
        float or None: The average of the max 'iteration-id' values across all files.
    """
    base_path = Path(base_dir)
    if use_sonia:
        pattern_str = rf"results-True-{re.escape(model_name)}-{lanes_count}-{vehicles_density}-\d+-spot"
    else:
        pattern_str = rf"results-True-{re.escape(model_name)}-{lanes_count}-{vehicles_density}-\d+"

    pattern = re.compile(pattern_str)

    max_iterations = []

    for folder in base_path.iterdir():
        if folder.is_dir() and pattern.fullmatch(folder.name):
            csv_file = folder / "evaluation.csv"
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    if "iteration-id" in df.columns:
                        max_iter = pd.to_numeric(df["iteration-id"], errors="coerce").max()
                        if pd.notna(max_iter):
                            if max_iter == 31:
                                max_iter = 30
                            print(csv_file, max_iter)

                            max_iterations.append(max_iter)
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")

    if max_iterations:
        return sum(max_iterations) / len(max_iterations)
    else:
        print("No valid 'iteration-id' found in any files.")
        return None

def verified_id_stats(csv_file):
    """
    Extracts and processes the 'verified-id' column from a CSV file.
    Returns (average of values != 3, percentage of values == 3)
    """
    try:
        df = pd.read_csv(csv_file)

        if "verified-id" in df.columns:
            values = df["verified-id"].dropna().astype(int)
            non_3_vals = values[values != 3]
            avg_not_3 = non_3_vals.mean() if not non_3_vals.empty else None

            percent_3 = (values == 3).sum() / len(values) * 100 if len(values) > 0 else None

            return avg_not_3, percent_3
        else:
            print(f"'verified-id' column not found in {csv_file}")
            return None, None

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None, None

def aggregate_verified_stats(base_dir, model_name, lanes_count, vehicles_density, use_sonia=False):
    base_path = Path(base_dir)
    if use_sonia:
        pattern_str = rf"results-True-{re.escape(model_name)}-{lanes_count}-{vehicles_density}-\d+-spot"
    else:
        pattern_str = rf"results-True-{re.escape(model_name)}-{lanes_count}-{vehicles_density}-\d+"

    pattern = re.compile(pattern_str)
    avg_list = []
    pct_list = []

    for folder in base_path.iterdir():
        if folder.is_dir() and pattern.fullmatch(folder.name):
            csv_file = folder / "evaluation.csv"
            if csv_file.exists():
                avg, pct = verified_id_stats(csv_file)
                if avg is not None:
                    avg_list.append(avg)
                if pct is not None:
                    pct_list.append(pct)

    result = {
        "average_verified_not_3": sum(avg_list) / len(avg_list) if avg_list else None,
        "percentage_verified_3": sum(pct_list) / len(pct_list) if pct_list else None,
    }
    return result


LANE_COUNT = 4
VEHICLE_DENSITY = 3.0
USE_SONIA = False

# ---- Call 1: Just Average Travelled ----
avg = average_travelled_value("./", "gpt-4o", LANE_COUNT, VEHICLE_DENSITY, use_sonia=USE_SONIA)
print(f"Average Travelled: {avg}")

avg_f = average_travelled_value_finished("./", "gpt-4o", LANE_COUNT, VEHICLE_DENSITY, use_sonia=USE_SONIA)
print(f"Average Travelled for the finished runs: {avg_f}")


# ---- Call 2: Verified-ID Stats ----
verified_stats = aggregate_verified_stats("./", "gpt-4o", LANE_COUNT, VEHICLE_DENSITY, use_sonia=USE_SONIA)
print(f"Average Verified-ID (not 3): {verified_stats['average_verified_not_3']}")
print(f"Percentage of Fail-safe: {verified_stats['percentage_verified_3']}%")

# ---- Call 3
plot_aggregate_verified_id_histogram(
    base_dir="./",
    model_name="gpt-4o",
    lanes_count=LANE_COUNT,
    vehicles_density=VEHICLE_DENSITY,
    use_sonia=USE_SONIA,
    save_path=f"aggregate_verified_id_histogram_{LANE_COUNT}_{VEHICLE_DENSITY}.png"
)

# ---- Call 4
avg_max_iter = average_max_iteration_id("./", "gpt-4o", LANE_COUNT, VEHICLE_DENSITY, use_sonia=USE_SONIA)
print(f"Average Max Iteration-ID: {avg_max_iter}")