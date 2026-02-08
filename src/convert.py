# Convert from .pkl to interaction file .inter
# Convert feature in .pkl to .npy

import os
import argparse
import pickle
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='book', help='name of datasets')

    args, _ = parser.parse_known_args()

    data_dir = os.path.join('data', args.dataset)
    train_path = os.path.join(data_dir, 'trn_mat.pkl')
    test_path = os.path.join(data_dir, 'tst_mat.pkl')
    valid_path = os.path.join(data_dir, 'val_mat.pkl')
    output_path = os.path.join(data_dir, f'{args.dataset}_cold.inter')
    assert os.path.exists(train_path), f"{train_path} does not exist."
    assert os.path.exists(test_path), f"{test_path} does not exist."    
    assert os.path.exists(valid_path), f"{valid_path} does not exist."

    if os.path.exists(data_dir) is False:
        os.makedirs(data_dir)

    train = pickle.load(open(train_path, 'rb'))
    test = pickle.load(open(test_path, 'rb'))
    valid = pickle.load(open(valid_path, 'rb'))
    # Convert various sparse matrix types to COO-like (row, col, data) arrays
    def to_coo_parts(mat):
        try:
            # scipy sparse
            import scipy.sparse as sps
            if isinstance(mat, sps.spmatrix):
                coo = mat.tocoo()
                return coo.row, coo.col, coo.data
        except Exception:
            pass

        # If it's a tuple/list of (row, col, data)
        if isinstance(mat, (list, tuple)) and len(mat) == 3:
            return mat[0], mat[1], mat[2]

        # If it's a dict with keys
        if isinstance(mat, dict):
            for kset in (('row', 'col', 'data'), ('rows', 'cols', 'data')):
                if all(k in mat for k in kset):
                    return mat[kset[0]], mat[kset[1]], mat[kset[2]]

        # If it's a numpy array (2D) convert nonzero entries
        try:
            if hasattr(mat, 'ndim') and mat.ndim == 2:
                rows, cols = mat.nonzero()
                data = mat[rows, cols]
                return rows, cols, data
        except Exception:
            pass

        raise ValueError('Unsupported matrix type for conversion to COO parts: %s' % type(mat))

    # Label mapping: train=0, valid=1, test=2
    parts = []
    for m, label in ((train, 0), (valid, 1), (test, 2)):
        rows, cols, data = to_coo_parts(m)
        parts.append((rows, cols, data, label))

    # load colduser list
    cold_user_path = os.path.join(data_dir, 'cold_users.pkl')
    if os.path.exists(cold_user_path):
        with open(cold_user_path, 'rb') as f:
            cold_users = pickle.load(f)
        print(f"Loaded {len(cold_users)} cold users from {cold_user_path}")
    print(cold_users[:10])  # Print first 10 cold users for verification


    # Write merged .inter file. Timestamp is the sequential index of the row (interaction index).
    with open(output_path, 'w', encoding='utf-8') as fout:
        fout.write('userID\titemID\trating\ttimestamp\tx_label\n')
        idx = 0
        for rows, cols, data, label in parts:
            # Ensure arrays are iterable and same length
            length = int(len(rows))
            for i in range(length):
                u = int(rows[i])
                v = int(cols[i])
                try:
                    r = float(data[i])
                except Exception:
                    # default rating when unavailable
                    r = 1.0
                if (label == 2) and (u in cold_users):
                    # Skip test interactions for cold users
                    continue
                fout.write(f'{u}\t{v}\t{r}\t{idx}\t{label}\n')
                idx += 1

    # remove cold in original test file




    # print(f'Merged interactions written to: {output_path}')
    
    # embed_path = os.path.join(data_dir, 'itm_emb_np.pkl')
    # assert os.path.exists(embed_path), f"{embed_path} does not exist."
    # with open(embed_path, 'rb') as f:
    #     item_emb = pickle.load(f)
    # # Convert item embeddings to numpy array and save as .npy
    # item_emb_np = np.array(item_emb)
    # npy_output_path = os.path.join(data_dir, 'text_feat.npy')
    # np.save(npy_output_path, item_emb_np)
    # print(f'Item embeddings saved to: {npy_output_path}')
    # numUser, numItem = train.shape[0], train.shape[1]
    # assert item_emb_np.shape[0] == numItem, "Item embedding count does not match number of items."
    # print('Conversion completed. {}, {} items, {} users.'.format(output_path, numItem, numUser))

    # user_profile_path = os.path.join(data_dir, 'usr_emb_np.pkl')
    # assert os.path.exists(user_profile_path), f"{user_profile_path} does not exist."
    # with open(user_profile_path, 'rb') as f:
    #     user_emb = pickle.load(f)
    # user_emb_np = np.array(user_emb)
    # user_npy_output_path = os.path.join(data_dir, 'user_feat.npy')
    # np.save(user_npy_output_path, user_emb_np)
    # print(f'User embeddings saved to: {user_npy_output_path}')
