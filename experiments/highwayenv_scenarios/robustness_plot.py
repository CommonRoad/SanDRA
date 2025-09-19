
#
# --- DiLu ---
# [-0.05       -0.05       -0.05       -0.05       -0.05       -0.05
#  -0.05       -0.05       -0.04003724]
#
# --- ours without rules ---
#
# [-0.05       -0.05       -0.05       -0.05       -0.03808246 -0.02264387
#  -0.02095708 -0.0185827  -0.01568336 -0.03047833 -0.04639459 -0.05
#  -0.01847619 -0.05       -0.05       -0.05        0.05        0.05
#  -0.05       -0.05       -0.05        0.02004762  0.05       -0.036
#   0.05200488  0.05352582  0.05        0.05        0.05        0.05
#   0.05      ]
#
# --- ours with rules ---
#
# [-0.05       -0.05       -0.04010302 -0.033256   -0.00544542 -0.02993467
#   0.          0.04914286  0.          0.06993551  0.07372805  0.0781697
#   0.08297526  0.08100538  0.08334522  0.0826853   0.07366681  0.06533441
#   0.05945031  0.05       -0.05       -0.05       -0.05827019  0.06731706
#   0.0641676   0.06226863 -0.05        0.06369537  0.05123444  0.05
#   0.05      ]

import matplotlib.pyplot as plt
import numpy as np

from experiments.vis_bar_rule_prompt import TUMcolor

# --- Data ---
dilu = np.array([
    -0.05, -0.05, -0.05, -0.05, -0.05, -0.05,
    -0.05, -0.05, -0.04003724
])

ours_no_rules = np.array([
    -0.05, -0.05, -0.05, -0.05, -0.03808246, -0.02264387,
    -0.02095708, -0.0185827, -0.01568336, -0.03047833, -0.04639459, -0.05,
    -0.01847619, -0.05, -0.05, -0.05, 0.05, 0.05,
    -0.05, -0.05, -0.05, 0.02004762, 0.05, -0.036,
    0.05200488, 0.05352582, 0.05, 0.05, 0.05, 0.05,
    0.05
])

ours_with_rules = np.array([
    -0.05, -0.05, -0.04010302, -0.033256, -0.00544542, -0.02993467,
    0., 0.04914286, 0., 0.06993551, 0.07372805, 0.0781697,
    0.08297526, 0.08100538, 0.08334522, 0.0826853, 0.07366681, 0.06533441,
    0.05945031, 0.05, -0.05, -0.05, -0.05827019, 0.06731706,
    0.0641676, 0.06226863, -0.05, 0.06369537, 0.05123444, 0.05,
    0.05
])

# --- Plotting ---
plt.figure(figsize=(8, 2.))

plt.plot(dilu, linestyle="-", label="DiLu", linewidth=2, color=TUMcolor.TUMblue.value)
plt.plot(ours_no_rules, linestyle="-", label="Ours (without rules)", linewidth=2, marker="", color=TUMcolor.TUMyellow.value)
plt.plot(ours_with_rules, linestyle="-", label="Ours (with rules)", linewidth=2, marker="",  color=TUMcolor.TUMgreen.value)


# Axis limits
plt.xlim(0, 30)
plt.ylim(-0.12, 0.12)
plt.yticks([-0.1, 0, 0.1])

# Grid lines for box-style visual
plt.grid(True, which='both', axis='x', linestyle='-', alpha=0.5)
plt.grid(True, which='both', axis='y', linestyle='-', alpha=0.7)


# Labels & title
plt.xlabel("Time step")
plt.ylabel("Robustness")
plt.title("Rule Robustness Comparison")

plt.legend()

plt.tight_layout()
plt.show()
