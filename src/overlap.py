import numpy as np

def topk_overlap(id_norm, sc_norm, k=10):
    """
    id_norm: np.ndarray of shape (n_users, dim)
    sc_norm: np.ndarray of shape (n_users, dim)
    Returns mean overlap count between top-k neighbors of idEmb and scEmb.
    """
    

    # # Normalize for cosine similarity
    # id_norm = id_emb / (np.linalg.norm(id_emb, axis=1, keepdims=True) + 1e-8)
    # sc_norm = sc_emb / (np.linalg.norm(sc_emb, axis=1, keepdims=True) + 1e-8)

    # Similarity matrices: (n_users, n_users)
    id_sim = id_norm @ id_norm.T
    sc_sim = sc_norm @ sc_norm.T

    # Zero out self-similarity (diagonal)
    np.fill_diagonal(id_sim, -np.inf)
    np.fill_diagonal(sc_sim, -np.inf)

    # Top-k indices for each user
    id_topk = np.argsort(id_sim, axis=1)[:, -k:]   # (n_users, k)
    sc_topk = np.argsort(sc_sim, axis=1)[:, -k:]   # (n_users, k)

    # Compute overlap for each user
    overlaps = np.array([
        len(np.intersect1d(id_topk[i], sc_topk[i]))
        for i in range(n_users)
    ])

    mean_overlap = overlaps.mean()
    return mean_overlap, overlaps

embeddings = np.load('movie.npy')  # Load your embeddings here
ss_emb = np.load('movie_ss.npy')  # Load your second set of embeddings here
tt_emb = np.load('user_feat.npy')  # Load your third set of embeddings here
n_users = embeddings.shape[0]
dim = embeddings.shape[1] // 2

sc_emb = embeddings[:, :dim]       # (n_users, dim)
id_emb = embeddings[:, dim:]       # (n_users, dim)
# id_sim = id_emb @ id_emb.T
sc_sim = sc_emb @ sc_emb.T
ss_sim = ss_emb @ ss_emb.T
# tt_sim = tt_emb @ tt_emb.T

# # check whether the two halves are different

# diff2 = np.linalg.norm(sc_sim - ss_sim, axis=1).mean()
# print(f'Mean L2 distance between ss_emb and tt_emb: {diff2:.4f}')



mean_overlap, overlaps = topk_overlap(id_emb, tt_emb, k=20)
print(f'Mean top-10 neighbor overlap: {mean_overlap:.4f}')

mean_overlap, overlaps = topk_overlap(id_emb, sc_emb, k=20)
print(f'Mean top-10 neighbor overlap: {mean_overlap:.4f}')

mean_overlap, overlaps = topk_overlap(id_emb, ss_emb, k=20)
print(f'Mean top-10 neighbor overlap: {mean_overlap:.4f}')
