"""
Compare All Classification Models
Generates a comprehensive comparison report
"""

import sys
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("COMPARING ALL CLASSIFICATION MODELS")
print("=" * 70)

# Load all model metrics
models = ['naive_bayes', 'svm', 'lstm', 'cnn']
results = {}

for model_name in models:
    metrics_path = Path(f"Outputs/results/{model_name}_test_metrics.json")
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            results[model_name] = json.load(f)
        print(f"Loaded metrics for {model_name}")
    else:
        print(f"Warning: Metrics not found for {model_name}")

# Create comparison DataFrame
comparison_data = []
for model_name, metrics in results.items():
    comparison_data.append({
        'Model': model_name.upper().replace('_', ' '),
        'Accuracy': metrics.get('accuracy', 0),
        'Precision': metrics.get('precision', 0),
        'Recall': metrics.get('recall', 0),
        'F1-Score': metrics.get('f1', 0),
        'AUC-ROC': metrics.get('auc', 0),
        'Top-k Accuracy': metrics.get('top_k_accuracy', 0)
    })

comparison_df = pd.DataFrame(comparison_data)

# Sort by F1-Score (best first)
comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

print("\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY")
print("=" * 70)
print(comparison_df.to_string(index=False))

# Save comparison table
output_path = Path("Outputs/results/model_comparison.csv")
comparison_df.to_csv(output_path, index=False)
print(f"\nComparison table saved to: {output_path}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# 1. Accuracy comparison
ax1 = axes[0, 0]
comparison_df.plot(x='Model', y='Accuracy', kind='barh', ax=ax1, color='skyblue')
ax1.set_xlabel('Accuracy')
ax1.set_title('Accuracy Comparison')
ax1.set_xlim(0, 1)

# 2. F1-Score comparison
ax2 = axes[0, 1]
comparison_df.plot(x='Model', y='F1-Score', kind='barh', ax=ax2, color='lightgreen')
ax2.set_xlabel('F1-Score')
ax2.set_title('F1-Score Comparison')
ax2.set_xlim(0, 1)

# 3. AUC-ROC comparison
ax3 = axes[1, 0]
comparison_df.plot(x='Model', y='AUC-ROC', kind='barh', ax=ax3, color='coral')
ax3.set_xlabel('AUC-ROC')
ax3.set_title('AUC-ROC Comparison')
ax3.set_xlim(0, 1)

# 4. All metrics radar-like comparison
ax4 = axes[1, 1]
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
x_pos = range(len(metrics_to_plot))
width = 0.2

for i, (idx, row) in enumerate(comparison_df.iterrows()):
    values = [row[m] for m in metrics_to_plot]
    ax4.bar([x + i*width for x in x_pos], values, width, label=row['Model'], alpha=0.8)

ax4.set_xlabel('Metrics')
ax4.set_ylabel('Score')
ax4.set_title('All Metrics Comparison')
ax4.set_xticks([x + width*1.5 for x in x_pos])
ax4.set_xticklabels(metrics_to_plot, rotation=45, ha='right')
ax4.legend()
ax4.set_ylim(0, 1)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_path = Path("Outputs/plots/model_comparison.png")
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Comparison plot saved to: {plot_path}")

# Generate text report
report_path = Path("Outputs/results/model_comparison_report.txt")
with open(report_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("RAHATAI - MODEL COMPARISON REPORT\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("SUMMARY:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total Models Evaluated: {len(results)}\n")
    f.write(f"Best Model (by F1-Score): {comparison_df.iloc[0]['Model']}\n")
    f.write(f"Best F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}\n")
    f.write(f"Best Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}\n\n")
    
    f.write("DETAILED COMPARISON:\n")
    f.write("-" * 70 + "\n")
    f.write(comparison_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("RANKING:\n")
    f.write("-" * 70 + "\n")
    for i, (idx, row) in enumerate(comparison_df.iterrows(), 1):
        f.write(f"{i}. {row['Model']}: F1={row['F1-Score']:.4f}, Acc={row['Accuracy']:.4f}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 70 + "\n")

print(f"Text report saved to: {report_path}")

print("\n" + "=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)
print(f"\nBest Model: {comparison_df.iloc[0]['Model']}")
print(f"  F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")
print(f"  Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}")
print(f"  AUC-ROC: {comparison_df.iloc[0]['AUC-ROC']:.4f}")

