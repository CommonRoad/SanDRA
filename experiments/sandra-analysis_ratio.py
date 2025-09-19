import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ----------------------------
# CONFIGURATION
# ----------------------------
MODEL_NAME = "gpt-4o"
BASE_DIR = Path("/home/liny/Documents/commonroad/results-sandra/new-results-updated")
LANE_DENSITY_PAIRS = [(4, 2.0), (4, 3.0), (5, 3.0)]
LONGITUDINAL_ACTIONS = ["accelerate", "decelerate", "keep"]
LATERAL_ACTIONS = ["follow_lane", "right", "left"]
ACTIONS = [(lon, lat) for lon in LONGITUDINAL_ACTIONS for lat in LATERAL_ACTIONS]
USE_RULES_IN_PROMPT_LIST = [True]
USE_RULES_IN_REACH_LIST = [True, False]
PREDICTION_TYPES = ["not-set-based", "set-based"]
SEEDS = [5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348, 4213, 2572]

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def map_to_high_level_action(lon_action, lat_action):
    lat_action_upper = lat_action.upper()
    lon_action_upper = lon_action.upper()
    if lat_action_upper == "LEFT":
        return "LANE_LEFT"
    elif lat_action_upper == "RIGHT":
        return "LANE_RIGHT"
    elif lat_action_upper == "FOLLOW_LANE":
        if lon_action_upper == "KEEP":
            return "IDLE"
        elif lon_action_upper == "ACCELERATE":
            return "FASTER"
        elif lon_action_upper == "DECELERATE":
            return "SLOWER"
    return "UNKNOWN"

def find_result_folders(base_dir, lane_count, vehicle_density, use_rules_in_prompt, use_rules_in_reach, prediction_type, seeds):
    """
    Find folders matching the configuration including seeds and prediction type.
    """
    matched_folders = []
    for seed in seeds:
        if prediction_type == "set-based":
            expected_pattern = (
                f"results-True-{MODEL_NAME}-{lane_count}-{vehicle_density}-{seed}"
                f"-spot-rule_prompt-{use_rules_in_prompt}-reach-{use_rules_in_reach}"
            )
        else:
            expected_pattern = (
                f"results-True-{MODEL_NAME}-{lane_count}-{vehicle_density}-{seed}"
                f"-rule_prompt-{use_rules_in_prompt}-reach-{use_rules_in_reach}"
            )

        for path in base_dir.rglob(expected_pattern):
            if path.is_dir():
                matched_folders.append(path)
    return matched_folders

def average_action_ratio(folder, lane_count, vehicle_density, target_lon, target_lat):
    """
    Compute normalized average ratio for a low-level action in one folder.
    """
    csv_file = folder / "evaluation.csv"
    if not csv_file.exists():
        return None

    try:
        df = pd.read_csv(csv_file)

        # Initialize counters for all actions
        action_counts = {(lon, lat): 0 for lon, lat in ACTIONS}
        total_valid_rows = 0

        # Iterate over each row
        for _, row in df.iterrows():
            verified_id = row.get("verified-id")
            if pd.isna(verified_id):
                continue
            try:
                idx = int(verified_id) + 1
            except ValueError:
                continue

            lateral_col = f"Lateral{idx}"
            longitudinal_col = f"Longitudinal{idx}"
            if lateral_col not in df.columns or longitudinal_col not in df.columns:
                continue

            lateral_val = row[lateral_col]
            longitudinal_val = row[longitudinal_col]
            if pd.isna(lateral_val) or pd.isna(longitudinal_val):
                continue

            total_valid_rows += 1

            # Increment count for the observed action
            key = (longitudinal_val, lateral_val)
            if key in action_counts:
                action_counts[key] += 1

        # Compute normalized ratios for all actions
        if total_valid_rows > 0:
            normalized_ratios = {action: count / total_valid_rows for action, count in action_counts.items()}
            return normalized_ratios.get((target_lon, target_lat), 0)
        else:
            return 0

    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None

# ----------------------------
# MAIN LOOP
# ----------------------------
overall_results = {}

for use_rules_in_prompt in USE_RULES_IN_PROMPT_LIST:
    for use_rules_in_reach in USE_RULES_IN_REACH_LIST:
        for prediction_type in PREDICTION_TYPES:
            config_key = (use_rules_in_prompt, use_rules_in_reach, prediction_type)
            print(f"\nProcessing config: {config_key}")

            # Aggregate storage
            average_action_rates = {}
            overall_high_level_sum = defaultdict(float)
            overall_high_level_count = defaultdict(int)

            high_level_rates_by_config_sum = defaultdict(lambda: defaultdict(float))
            high_level_rates_by_config_count = defaultdict(lambda: defaultdict(int))

            for lane_count, vehicle_density in LANE_DENSITY_PAIRS:
                matched_folders = find_result_folders(
                    BASE_DIR, lane_count, vehicle_density, use_rules_in_prompt, use_rules_in_reach, prediction_type, SEEDS
                )
                if not matched_folders:
                    print(f"No folders found for lane={lane_count}, density={vehicle_density}")
                    continue

                for (target_lon, target_lat) in ACTIONS:
                    action_rates = []
                    for folder in matched_folders:
                        rate = average_action_ratio(folder, lane_count, vehicle_density, target_lon, target_lat)
                        if rate is not None:
                            action_rates.append(rate)
                            high_action = map_to_high_level_action(target_lon, target_lat)
                            config = (lane_count, vehicle_density)
                            high_level_rates_by_config_sum[config][high_action] += rate
                            high_level_rates_by_config_count[config][high_action] += 1

                    avg_rate = sum(action_rates) / len(action_rates) if action_rates else float('nan')
                    average_action_rates[(lane_count, vehicle_density, target_lon, target_lat)] = avg_rate

                    if not np.isnan(avg_rate):
                        high_action = map_to_high_level_action(target_lon, target_lat)
                        overall_high_level_sum[high_action] += avg_rate
                        overall_high_level_count[high_action] += 1

            # Compute overall high-level averages
            overall_high_level_avg = {}
            total_sum = sum(overall_high_level_sum.values())
            if total_sum > 0:
                overall_high_level_avg = {action: val / total_sum for action, val in overall_high_level_sum.items()}

            # Compute per-config high-level averages
            high_level_rates_by_config_avg = {}
            for config in high_level_rates_by_config_sum:
                high_level_rates_by_config_avg[config] = {}
                total_config_sum = sum(high_level_rates_by_config_sum[config].values())
                if total_config_sum > 0:
                    for action, val in high_level_rates_by_config_sum[config].items():
                        high_level_rates_by_config_avg[config][action] = val / total_config_sum

            overall_results[config_key] = {
                "low_level": average_action_rates,
                "high_level_overall": overall_high_level_avg,
                "high_level_per_config": high_level_rates_by_config_avg
            }

# ----------------------------
# PRINT RESULTS
# ----------------------------
for config, results in overall_results.items():
    print(f"\n=== Config: {config} ===")
    print("Low-level action averages:")
    for key, rate in results["low_level"].items():
        print(f"Lane={key[0]}, Density={key[1]}, Action=({key[2]},{key[3]}): {rate:.4f}")

    print("\nOverall high-level action averages (normalized):")
    for action, rate in results["high_level_overall"].items():
        print(f"{action}: {rate:.4f}")

    print("\nPer-configuration high-level averages (normalized):")
    for config_key, action_dict in results["high_level_per_config"].items():
        print(f"Lane={config_key[0]}, Density={config_key[1]}")
        for action, rate in action_dict.items():
            print(f"  {action}: {rate:.4f}")
