
import torch
from sklearn.metrics.pairwise import cosine_similarity

def build_affinity_matrix(feature_tensor):
    T, P, D = feature_tensor.shape
    flat = feature_tensor.view(T * P, D).numpy()
    A = cosine_similarity(flat)
    return A  # [TP x TP]
