import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix


def threshold_cost_curve(
    y_true,
    y_prob,
    cost_fn=500,
    cost_fp=50,
    out_path="static/plots/threshold_vs_cost.png"
):
    thresholds = np.arange(0.01, 1.0, 0.01)
    total_costs = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        cost = (fn * cost_fn) + (fp * cost_fp)
        total_costs.append(cost)

    # Plot
    plt.figure()
    plt.plot(thresholds, total_costs)
    plt.xlabel("Threshold")
    plt.ylabel("Total Business Cost")
    plt.title("Threshold vs Business Cost")
    plt.grid(True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

    # Best threshold
    best_idx = np.argmin(total_costs)
    best_threshold = thresholds[best_idx]

    return best_threshold, thresholds, total_costs
