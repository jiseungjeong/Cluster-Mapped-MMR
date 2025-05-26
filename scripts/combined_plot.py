import matplotlib.pyplot as plt
import numpy as np

# Accuracy data (from accuracy_plot.py)
methods = [
    "zero-shot",
    "random",
    "similiarity",
    "MMR",
    "k-means",
    "HDBSCAN",
    "CM-MMR\n(k-means)",
    "CM-MMR\n(HDBSCAN)",
]
accuracies = [
    84.17,
    83.1,
    83.4,
    85.73,
    83.34,
    84.1,
    85.53,
    86.33,
]

# Latency data (from latency_comparison_plot.py)
# Converting seconds to milliseconds (multiply by 1000)
latencies = [
    1.659e-05 * 1000,  # Zero-shot
    2.55e-05 * 1000,  # random
    0.002015 * 1000,  # similarity
    0.0075 * 1000,  # MMR
    8.9e-05 * 1000,  # k-means
    0.00026 * 1000,  # HDBSCAN
    0.0022 * 1000,  # CM-MMR(k-means)
    0.0021 * 1000,  # CM-MMR(HDBSCAN)
]

# Create figure with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

# Plot accuracy bars on the left axis
bars = ax1.bar(methods, accuracies, color="lightblue", alpha=0.7)
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(80, 90)

# Plot latency line on the right axis
ax2.plot(methods, latencies, color="red", marker="o", linestyle="-", linewidth=2)
ax2.set_ylabel("Latency (ms)")
ax2.set_ylim(0, max(latencies) * 1.3)

# Add accuracy values on top of bars
for bar in bars:
    yval = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 0.3,
        f"{yval:.1f}%",
        ha="center",
        va="bottom",
    )

# Add latency values on line points
for i, latency in enumerate(latencies):
    ax2.text(
        i,
        latency + max(latencies) * 0.05,
        f"{latency:.3f}",
        ha="center",
        va="bottom",
        color="red",
        fontsize=8,
    )

# Set title and adjust layout
plt.title("Accuracy and Latency Comparison between Methods")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.xticks(rotation=15, fontsize=6)
plt.tight_layout()

# Add legend
ax1.legend(["Accuracy"], loc="upper left")
ax2.legend(["Latency"], loc="upper right")

plt.savefig("combined_accuracy_latency.png", dpi=300, bbox_inches="tight")
plt.show()
