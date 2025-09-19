import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

class TUMcolor(tuple, Enum):
    TUMblue = (0, 101 / 255, 189 / 255)
    TUMgreen = (162 / 255, 173 / 255, 0)
    TUMyellow = (254 / 255, 215 / 255, 2 / 255)
    TUMred = ((234 / 255, 114 / 255, 55 / 255))
    TUMgray = (156 / 255, 157 / 255, 159 / 255)


# Labels
labels = ['Safe_Top1', 'Safe_TopK', 'Match_Top1', 'Match_TopK']

# Data
with_rules = [92.38, 99.75, 29.84, 82.15]
without_rules = [89.62, 99.75, 34.62, 74.75]

# Bar settings
x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(8, 2.5))

# Bars
bars1 = ax.bar(x - width/2, without_rules, width, label='Without rules', color=TUMcolor.TUMyellow.value)
bars2 = ax.bar(x + width/2, with_rules, width, label='With rules', color=TUMcolor.TUMgreen.value)

# Axis settings
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 110)
ax.set_yticks([0, 50, 100])
plt.xlim(left=-0.5, right=len(labels) - 0.5 )

# Grid lines for box-style visual
ax.grid(True, which='both', axis='x', linestyle='-', alpha=0.5)
ax.grid(True, which='both', axis='y', linestyle='-', alpha=0.7)


# Legend
ax.legend(loc='upper right', frameon=True)

# Add labels above bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(bars1)
autolabel(bars2)

# Layout
plt.tight_layout()
plt.show()
