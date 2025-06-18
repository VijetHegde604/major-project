import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSVs
pred_df = pd.read_csv("results/predictions.csv")
gt_df = pd.read_csv("results/ground_truth.csv")

# Merge on image_name
merged = pd.merge(gt_df, pred_df, on="image_name")
y_true = merged["true_label"]
y_pred = merged["predicted_label"]

# 🔢 Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"✅ Accuracy: {accuracy * 100:.2f}%")

# 📊 Precision, Recall, F1-score
print("\n🔍 Classification Report:")
print(classification_report(y_true, y_pred, digits=3))

# 🌀 Confusion Matrix
labels = sorted(y_true.unique())
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
