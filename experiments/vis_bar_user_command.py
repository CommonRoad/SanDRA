import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

class TUMcolor(tuple, Enum):
    TUMblue = (0, 101 / 255, 189 / 255)
    TUMgreen = (162 / 255, 173 / 255, 0)
    TUMyellow = (254 / 255, 215 / 255, 2 / 255)

# Data
labels = ['Safe_Top1', 'Safe_TopK', 'Match_Top1', 'Match_TopK']
aggressive = [77.25, 99.88, 7.5, 55.5]
no_command = [89.62, 99.75, 34.62, 74.75]
cautious = [91.62, 99.75, 38.75, 78.75]

print(np.mean([aggressive, no_command, cautious], axis=0))

# Prepare bar locations
x = np.arange(len(labels))
width = 0.15

fig, ax = plt.subplots(figsize=(8, 2.5))

# Bar plots
bars1 = ax.bar(x - width, aggressive, width, label='Drive aggressively', color=TUMcolor.TUMblue.value)
bars2 = ax.bar(x, no_command, width, label='No command', color=TUMcolor.TUMgreen.value)
bars3 = ax.bar(x + width, cautious, width, label='Drive cautiously', color=TUMcolor.TUMyellow.value)

# Axis settings
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.ylim(0, 110)
plt.xlim(left=-0.5, right=len(labels) - 0.5 )
plt.ylabel('')  # Remove y-axis label

# Grid lines for box-style visual
ax.grid(True, which='both', axis='x', linestyle='-', alpha=0.5)
ax.grid(True, which='both', axis='y', linestyle='-', alpha=0.7)

# Legend
plt.legend(loc='upper right', frameon=True)

# Add text labels on top of bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
#
# autolabel(bars1)
# autolabel(bars2)
# autolabel(bars3)

# Tight layout
plt.tight_layout()
plt.show()
