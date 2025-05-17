from sklearn.neighbors import NearestNeighbors
import torch

def build_edge_index(X, k=5):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    edges = []
    for i, neighbors in enumerate(nbrs.kneighbors(X, return_distance=False)):
        for j in neighbors:
            if i != j:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index