import matplotlib.pyplot as plt
import numpy as np
from enum import Enum



# GPT-4o
#
# Safe_Top1: 89.62%
# Safe_TopK: 99.75%
# Match_Top1: 34.62%
# Match_TopK: 74.75%
#
#
# Finetuned GPT-4o
# Safe_Top1: 99.62%
# Safe_TopK: 100.00%
# Match_Top1: 61.38%
# Match_TopK: 80.75%
#
# Qwen3-0.6B
#
# Safe_Top1: 74.50%
# Safe_TopK: 91.88%
# Match_Top1: 32.12%
# Match_TopK: 49.38%

# Finetuned Qwen3-0.6B
#
# Safe_Top1: 99.38%
# Safe_TopK: 99.88%
# Match_Top1: 53.25%
# Match_TopK: 78.62%


class TUMcolor(tuple, Enum):
    TUMblue = (0, 101 / 255, 189 / 255)
    TUMlightblue = (100/255, 160/255, 200/255)
    TUMgreen = (162 / 255, 173 / 255, 0)
    TUMyellow = (254 / 255, 230 / 255, 103 / 255)
    TUMgrey = (153 / 255, 153 / 255, 153 / 255)
    TUMred = (227 / 255, 27 / 255, 35 / 255)
    TUMdarkred = (139 / 255, 0, 0)
    TUMgray = (156 / 255, 157 / 255, 159 / 255)
    TUMdarkgray = (88 / 255, 88 / 255, 99 / 255)
    TUMorange = (227 / 255, 114 / 255, 34 / 255)
    TUMdarkblue = (0, 82 / 255, 147 / 255)
    TUMwhite = (1, 1, 1)
    TUMblack = (0, 0, 0)
    TUMlightgray = (217 / 255, 218 / 255, 219 / 255)
    TUMdarkyellow = (203 / 255, 171 / 255, 1 / 255)

# Labels
labels = ['Safe_Top1', 'Safe_TopK', 'Match_Top1', 'Match_TopK']
x = np.arange(len(labels))
width = 0.15

# Data
gpt4o = [89.62, 99.75, 34.62, 74.75]
gpt4o_ft = [99.62, 100.0, 61.38, 80.75]
qwen = [74.50, 91.88, 32.12, 49.38]
qwen_ft = [99.38, 99.88, 53.25, 78.62]


# Plot
fig, ax = plt.subplots(figsize=(8, 2.5))

bars1 = ax.bar(x - 1.5 * width, gpt4o, width, label='GPT-4o', color=TUMcolor.TUMlightblue.value)
bars2 = ax.bar(x - 0.5 * width, gpt4o_ft, width, label='Finetuned GPT-4o', color=TUMcolor.TUMdarkblue.value)
bars3 = ax.bar(x + 0.5 * width, qwen, width, label='Qwen3-8B', color=TUMcolor.TUMyellow.value)
bars4 = ax.bar(x + 1.5 * width, qwen_ft, width, label='Finetuned Qwen3-0.6B', color=TUMcolor.TUMdarkyellow.value)

# Axis
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.ylim(0, 110)
ax.set_yticks([0, 50, 100])
# plt.ylabel('Performance (%)')
# ax.set_title('Performance Comparison of Language Models')

# Grid
# Grid lines for box-style visual
ax.grid(True, which='both', axis='x', linestyle='-', alpha=0.5)
ax.grid(True, which='both', axis='y', linestyle='-', alpha=0.7)

# Annotate bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# for bars in [bars1, bars2, bars3, bars4]:
#     autolabel(bars)

# Legend
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)

plt.tight_layout()
plt.show()