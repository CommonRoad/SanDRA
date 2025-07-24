import os
import pandas as pd

from sandra.common.config import SanDRAConfiguration


def evaluate_highenv_results():
    results_dir = "results-True-gpt-4o/"

    max_iterations = []
    verifier_fails = []

    for filename in os.listdir(results_dir):
        if not filename.startswith('new'):
            continue
        result_df = pd.read_csv(results_dir + filename)

        max_iteration = 0
        verifier_fail = 0
        counter = 0
        for idx, row in result_df.iterrows():
            if max_iteration < int(row['iteration-id']):
                max_iteration = int(row['iteration-id'])
            if int(row["verified-id"]) > 2:
                verifier_fail += 1
            counter += 1
        max_iterations.append(max_iteration)
        verifier_fails.append(verifier_fail) # / counter)

    average_max_iterations = sum(max_iterations) / len(max_iterations)
    average_verifier_fails = sum(verifier_fails) # / len(verifier_fails)
    passed = [x > 30 for x in max_iterations]
    average_passed = sum(passed) / len(passed)
    average_passed *= 100

    print(f"Average Max Iterations: {average_max_iterations:.1f}")
    print(f"Sum Verifier Fails: {average_verifier_fails:.1f}")
    print(f"Average Passed: {average_passed:.1f}")


def evaluate_highd_results():
    k = SanDRAConfiguration().k
    results_dir = "results-highD-ft:gpt-4o-2024-08-06:tum::BsuinSqR/"
    labels_dir = "../highd_scenarios/batch_labelling_results_100.csv"
    labels_df = pd.read_csv(labels_dir)
    save_at = []
    match_at = []

    for filename in os.listdir(results_dir):
        try:
            result_df = pd.read_csv(results_dir + filename).iloc[0]
        except IndexError:
            print(f"Skipping scenario {filename}")
            raise ValueError()
        scenario_id = filename[:-4]
        matching_rows = labels_df[labels_df['ScenarioID'] == scenario_id]

        verified_id = result_df['verified-id']
        save_at.append(verified_id + 1)
        if verified_id >= k:
            match_at.append(scenario_id)
            continue

        if len(matching_rows) == 1:
            row = matching_rows.iloc[0]
        else:
            raise ValueError(f"Found {len(matching_rows)} matches for scenario {scenario_id}")
        lateral_label = row["Trajectory_Lateral"]
        longitudinal_label = row["Trajectory_Longitudinal"]
        print(len(match_at))
        found_none = True
        for i in range(k):
            lateral = result_df[f"Lateral{i+1}"]
            longitual = result_df[f"Longitudinal{i+1}"]
            if lateral == lateral_label and longitual == longitudinal_label:
                match_at.append(i + 1)
                found_none = False
                break
        if found_none:
            match_at.append(i + 1)

    save_at_1 = [x == 1 for x in save_at]
    save_at_1 = (sum(save_at_1) / len(save_at_1)) * 100
    save_at_3 = [x <= 3 for x in save_at]
    save_at_3 = (sum(save_at_3) / len(save_at_3)) * 100
    print(f"Save@1: {save_at_1:.1f}%")
    print(f"Save@3: {save_at_3:.1f}%")

    match_at_1 = [x == 1 for x in match_at]
    match_at_1 = (sum(match_at_1) / len(match_at_1)) * 100
    match_at_3 = [x <= 3 for x in match_at]
    match_at_3 = (sum(match_at_3) / len(match_at_3)) * 100
    match_at_5 = [x <= 5 for x in match_at]
    match_at_5 = (sum(match_at_5) / len(match_at_5)) * 100

    print(f"Match@1: {match_at_1:.1f}%")
    print(f"Match@3: {match_at_3:.1f}%")
    print(f"Match@5: {match_at_5:.1f}%")


if __name__ == '__main__':
    evaluate_highd_results()