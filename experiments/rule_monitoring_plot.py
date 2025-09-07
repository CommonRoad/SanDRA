import matplotlib.pyplot as plt
import numpy as np

# Configurations and labels
labels = [
    'Prompt✓, Reach✓',
    'Prompt✗, Reach✓',
    'Prompt✓, Reach✗',
    'Prompt✗, Reach✗'
]
x = np.arange(len(labels))
width = 0.35

# Violation data
set_based = {
    "R_G1": [0.119, 0.122, 0.309, 0.294],
    "R_G2": [0.248, 0.252, 0.228, 0.250],
    "R_G3": [0.000, 0.000, 0.000, 0.000],
}
most_likely = {
    "R_G1": [0.119, 0.126, 0.321, 0.329],
    "R_G2": [0.245, 0.264, 0.232, 0.236],
    "R_G3": [0.000, 0.000, 0.000, 0.000],
}

# Convert to satisfaction rate
set_based_sat = {rule: [1 - v for v in vals] for rule, vals in set_based.items()}
most_likely_sat = {rule: [1 - v for v in vals] for rule, vals in most_likely.items()}

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle('Satisfaction Rates Across Rules', fontsize=16, fontweight='bold')

rules = ["R_G1", "R_G2", "R_G3"]
colors = ['#1f77b4', '#ff7f0e']

for i, rule in enumerate(rules):
    ax = axes[i]
    bars1 = ax.bar(x - width/2, set_based_sat[rule], width, label='Set-based',
                   color=colors[0], edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, most_likely_sat[rule], width, label='Most-likely',
                   color=colors[1], edgecolor='black', alpha=0.8)

    ax.set_title(rule, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

axes[0].set_ylabel('Satisfaction Rate', fontsize=12)
fig.legend(title="Prediction Type", loc='upper center', ncol=2, fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
