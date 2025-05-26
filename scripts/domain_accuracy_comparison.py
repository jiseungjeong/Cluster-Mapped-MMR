import matplotlib.pyplot as plt
import numpy as np

# Dataset names and accuracy values
datasets = ["ARC", "CommonsenseQA", "GSM8K", "Combined"]
accuracies = [0.96, 0.77, 0.89, 0.8133]

# Set up the figure and axis
plt.figure(figsize=(10, 6))

# Create colors - single domain datasets in one color, combined in another
colors = ["#4CAF50", "#4CAF50", "#4CAF50", "#FF5722"]

# Create the bar chart
bars = plt.bar(datasets, accuracies, color=colors, width=0.6)

# Add a horizontal line showing the combined accuracy
plt.axhline(
    y=accuracies[3],
    color="#FF5722",
    linestyle="--",
    alpha=0.7,
    label=f"Combined Accuracy: {accuracies[3]:.2f}",
)

# Add percentage labels on top of each bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.01,
        f"{height:.0%}",
        ha="center",
        va="bottom",
        fontsize=11,
    )

# Add titles and labels
plt.title("Accuracy Comparison Across Different Domains", fontsize=14)
plt.xlabel("Dataset Domain", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0, 1.05)  # Set y-axis limits

# Add a legend
plt.legend(loc="lower right")

# Add grid lines for better readability
plt.grid(axis="y", linestyle="--", alpha=0.3)

# Add annotation explaining the finding
plt.annotate(
    "The combined dataset shows lower accuracy\ncompared to ARC and GSM8K domains",
    xy=(3, accuracies[3]),
    xytext=(2.5, 0.5),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
    fontsize=10,
    ha="center",
)

# Tight layout
plt.tight_layout()

# Save the figure
plt.savefig("domain_accuracy_comparison.png", dpi=300)
plt.savefig("domain_accuracy_comparison.pdf")

print(
    "Graph saved as domain_accuracy_comparison.png and domain_accuracy_comparison.pdf"
)

# Show the figure
plt.show()
