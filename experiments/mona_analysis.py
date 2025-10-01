import pandas as pd

if __name__ == "__main__":
    # For example:
    batch_file = "./data/rule_prompt/batch_labelling_results_gpt-4o_20250912_123945-rule_False.csv"
    df = pd.read_csv(batch_file)

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