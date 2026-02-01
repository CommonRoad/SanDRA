import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

class TUMcolor(tuple, Enum):
    TUMblue = (0, 101 / 255, 189 / 255)
    TUMgreen = (162 / 255, 173 / 255, 0)
    TUMyellow = (254 / 255, 215 / 255, 2 / 255)

# Data
labels = ['Safe_Top1', 'Safe_TopK', 'Match_Top1', 'Match_TopK']
t50 = [97.11, 98.69, 32.98, 76.74]
t25 = [89.62, 99.75, 34.62, 74.75]
t10 = [81.80, 96.71, 33.79, 77.79]

print(np.mean([t10, t25, t50], axis=0))

# Prepare bar locations
x = np.arange(len(labels))
width = 0.15

fig, ax = plt.subplots(figsize=(8, 2.5))

# Bar plots
bars1 = ax.bar(x - width, t10, width, label='t10', color=TUMcolor.TUMgreen.value, alpha=0.33)
bars2 = ax.bar(x, t25, width, label='t25', color=TUMcolor.TUMgreen.value, alpha=0.66)
bars3 = ax.bar(x + width, t50, width, label='t50', color=TUMcolor.TUMgreen.value)

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
