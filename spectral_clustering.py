
from sklearn.cluster import SpectralClustering

def segment_with_spectral(A, n_clusters):
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
    labels = clustering.fit_predict(A)
    return labels
