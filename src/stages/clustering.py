import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_with_kmeans(
    X_embs: np.array,
    min_k: int = 10, max_k: int = 100,
    seed: int = 0,
):
    # Choose k with max silhouette value
    silhouettes = []

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        kmeans.fit(X_embs)
        y_kmeans = kmeans.predict(X_embs)
        
        silhouette = silhouette_score(X_embs, y_kmeans)
        silhouettes.append(silhouette)

    k_val = np.argmax(np.array(silhouettes)) + min_k

    # cluster for optimal k
    kmeans = KMeans(n_clusters=k_val, random_state=seed, n_init="auto")
    kmeans.fit(X_embs)
    y_kmeans = kmeans.predict(X_embs)

    # save list with clusters elems ids lists
    cluster_id_to_elems = []

    for cluster_id in range(k_val):
        cluster_elems_ids = np.argwhere(y_kmeans == cluster_id).reshape(-1).tolist()
        cluster_id_to_elems.append(cluster_elems_ids)

    return cluster_id_to_elems
