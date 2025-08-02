import pandas as pd

if __name__ == "__main__":
    prefix = "/home/sebastian/Documents/Uni/Sandra/mona_scenarios/"
    batch_file = "batch_labelling_results_qwen3-0.6b:latest_20250801_155446.csv"
    batch_file = "batch_labelling_results_qwen3-0.6B-highD:latest_20250801_162507.csv"
    batch_file = "batch_labelling_results_qwen3-0.6B-16-highD:latest_20250801_165416.csv"
    #batch_file = "/home/sebastian/Documents/Uni/Sandra/mona_scenarios/batch_labelling_results_qwen3-0.6b-16-3-highD:latest_20250801_105403.csv"
    df = pd.read_csv(prefix + batch_file)

    # Columns to compute average TRUE rate
    target_columns = ['Safe_Top1', 'Safe_TopK', 'Match_Top1', 'Match_TopK']

    # Ensure values are boolean (in case they are strings like "TRUE"/"FALSE")
    df[target_columns] = df[target_columns].applymap(
        lambda x: str(x).strip().lower() in ['true', '1', 'yes']
    )

    # Compute average TRUE rate for each column
    true_rates = df[target_columns].mean()

    # Display result as percentages
    print("Average TRUE rates (%):")
    for col, rate in true_rates.items():
        print(f"{col}: {rate * 100:.2f}%")