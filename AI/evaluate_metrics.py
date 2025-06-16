import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

# Load feature vectors and labels
gallery_vectors = np.load("gallery_vectors.npy")
gallery_labels = np.load("gallery_labels.npy")

query_vectors = np.load("query_vectors.npy")
query_labels = np.load("query_labels.npy")

def evaluate(query_vectors, query_labels, gallery_vectors, gallery_labels, k=5):
    top1_correct = 0
    topk_correct = 0
    total_map = 0

    for qv, ql in zip(query_vectors, query_labels):
        sims = cosine_similarity([qv], gallery_vectors)[0]
        idx = np.argsort(sims)[::-1][:k]
        pred_labels = gallery_labels[idx]

        if pred_labels[0] == ql:
            top1_correct += 1
        if ql in pred_labels:
            topk_correct += 1

        relevance = (gallery_labels == ql).astype(int)
        total_map += average_precision_score(relevance, sims)

    total = len(query_labels)
    top1 = top1_correct / total
    topk = topk_correct / total
    mAP = total_map / total
    return top1, topk, mAP

# Evaluate and print
top1, top5, map_score = evaluate(query_vectors, query_labels, gallery_vectors, gallery_labels, k=5)

print(f"Top-1 Accuracy: {top1:.4f}")
print(f"Top-5 Accuracy: {top5:.4f}")
print(f"Mean Average Precision (mAP): {map_score:.4f}")
