import torch
from sklearn.metrics.pairwise import cosine_similarity

def build_edge_index(X, threshold):
    similarity_matrix = cosine_similarity(X)
    edges = []
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            if i != j and similarity_matrix[i][j] > threshold:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index