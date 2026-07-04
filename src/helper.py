import json

from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
bertmodel = SentenceTransformer('all-MiniLM-L6-v2')
import numpy as np
# Load json files, encoding='utf-8'

def merge_json_files(path1, path2, output_path):
	with open(path1, 'r', encoding='utf-8') as f:
		p1 = json.load(f)
	with open(path2, 'r', encoding='utf-8') as f:
		p2 = json.load(f)

	merged = {**p1, **p2}

	with open(output_path, 'w') as f:
		json.dump(merged, f, indent=2)

	print("Files merged successfully into merged.json")

	# print total number of entries
	print(f"Total number of entries in merged file: {len(merged)}")
	return merged

def get_profile_text(files):
	counter = 0
	profile = []
	for ii in range(len(files)):
		try:
			userProfile = json.loads(files[str(ii)]['summary'])['summarization']
			profile.append(userProfile)
			counter += 1
		except Exception:
			print(f"Error parsing JSON for key: {ii}")
			print(f"Value: {files[str(ii)]}")
	print(f"Total successfully parsed profiles: {counter}")
	return profile

def get_profile_embeddings(inputs, save=False, path=None):
	embeddings = np.array(bertmodel.encode(inputs, show_progress_bar=True))	
	if save and path:
		np.save(path, embeddings)
	print("shape of embeddings:", embeddings.shape)  # should be (num_row, embedding_dim)
	return embeddings

def build_item_item_knn(itemDesc, top_k=50, metric="cosine", return_scores=False):
    """
    itemDesc: np.ndarray shape (num_items, dim) hoặc list-of-embeddings
    top_k: số hàng xóm gần nhất (không tính chính nó)
    metric: "cosine" (khuyến nghị) hoặc "euclidean"
    return_scores: nếu True trả thêm độ tương đồng/ khoảng cách
    """
    X = np.asarray(itemDesc, dtype=np.float32)
    n = X.shape[0]
    if top_k >= n:
        raise ValueError(f"top_k must be < num_items (top_k={top_k}, num_items={n})")

    # Với cosine: sklearn trả về "cosine distance" = 1 - cosine_similarity
    nn = NearestNeighbors(n_neighbors=top_k + 1, metric=metric, algorithm="auto")
    nn.fit(X)

    _, indices = nn.kneighbors(X, return_distance=True)  # (n, top_k+1)

    # loại bỏ chính nó (thường ở vị trí 0)
    item_item = indices[:, 1:top_k+1].astype(np.int32)

    if not return_scores:
        return item_item
	
def getUser_Interaction(interDF):
	user_interactions = {}
	for idx, row in tqdm(interDF.iterrows(), total=interDF.shape[0]):
		uid = row['userID']
		iid = row['itemID']
		label = row['x_label']
		if label != 0:
			continue
		if uid not in user_interactions:
			user_interactions[uid] = []
		user_interactions[uid].append(int(iid))
	return user_interactions

def get_itemDesc(metaDF, merge=True):
	itemDesc = []
	for idx, row in tqdm(metaDF.iterrows(), total=metaDF.shape[0]):
		# iid = row['iid']
		title = row['title']
		description = row['profile']
		if merge:
			itemDesc.append(f"Title: {title}\nDescription: {description}\n\n")
		else:
			itemDesc.append((title, description))
	return itemDesc


if __name__ == "__main__":
	pass
	# merge_json_files("./data/book/usr_prf_0.json", "./data/book/usr_prf_1.json", "./data/book/usr0.json")
	# merge_json_files("./data/book/usr_prf_2.json", "./data/book/usr_prf_3.json", "./data/book/usr1.json")
	# merge_json_files("./data/book/usr0.json", "./data/book/usr1.json", "./data/book/tuning_user_profile.json")