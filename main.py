
from extract_frames import extract_frames
from dino_features import extract_patch_features
from build_affinity import build_affinity_matrix
from spectral_clustering import segment_with_spectral
from visualize import visualize_clusters

# Step 1: Extract frames
extract_frames("input_video.mp4", "frames")

# Step 2: Extract ViT patch features
features = extract_patch_features("frames")  # [T, P, D]

# Step 3: Build affinity matrix
A = build_affinity_matrix(features)

# Step 4: Spectral Clustering
labels = segment_with_spectral(A, n_clusters=5)

# Step 5: Visualize
visualize_clusters(labels, "frames")
