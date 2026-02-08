import pickle
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import random
import numpy as np
import os
import json
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
def getTop100SimilarUsers(user_id, user_embeddings, user_ids, topk = 100):
	# user_embeddings is numpy array of shape (num_users, embedding_dim)
	userembed = user_embeddings[user_id]
	similarities = []
	for other_user_id in user_ids:
		if other_user_id == user_id:
			continue
		other_user_embed = user_embeddings[other_user_id]
		# cosine_similarity = torch.nn.functional.cosine_similarity(userembed, other_user_embed, dim=0)
		cosine_similarity = np.dot(userembed, other_user_embed) / (np.linalg.norm(userembed) * np.linalg.norm(other_user_embed))
		similarities.append((other_user_id, cosine_similarity))

	similarities.sort(key=lambda x: x[1], reverse=True)
	top_k_similar_users = similarities[:topk]
	return top_k_similar_users

def countOverlap(user_interactions, user_id, similar_user_ids):
	user_items = set(user_interactions[user_id])
	overlap_counts = {}
	for sim_user_id in similar_user_ids:
		sim_user_items = set(user_interactions[sim_user_id])
		overlap = len(user_items.intersection(sim_user_items))
		overlap_counts[sim_user_id] = overlap
	return overlap_counts

def user2info(user_id, user_interactions, item_profile):
	interacted_items = user_interactions[user_id]
	interacted_item_profiles = item_profile[item_profile['iid'].isin(interacted_items)]
    #  print the title, description and profile of the interacted items
	for idx, row in interacted_item_profiles.iterrows():
		print(f"Item ID: {row['iid']}")
		print(f"Title: {row['title']}")
		print(f"Description: {row['description']}")
		print(f"Profile: {row['profile']}")
		print("-" * 50)

dataset = "book"
file_path = f'./data/{dataset}/{dataset}.inter'
interDF = pd.read_csv(file_path, sep="\t", usecols=['userID', 'itemID', 'x_label'])
interDF['userID'] = interDF['userID'].astype(int)
interDF['itemID'] = interDF['itemID'].astype(int)

user_interactions = getUser_Interaction(interDF)
# # random embedding for each user of shape (embedding_dim,)
embedding_dim = 64
user_ids = list(user_interactions.keys())
coldEmPath = "CoLD_user_embeddings.npy"
path = f'./data/{dataset}/{coldEmPath}'
if os.path.exists(path):
	print(f"{path} exists, load user embedding.")
	coldEm = np.load(path, allow_pickle=True)
	print(f"Loaded user embeddings for users, shape: {coldEm.shape}")

rlmEmPath = "rlm_plus_user_embeddings.npy"
path_rlm = f'./data/{dataset}/{rlmEmPath}'
if os.path.exists(path_rlm):
	print(f"{path_rlm} exists, load user embedding.")
	rlmEm = np.load(path_rlm, allow_pickle=True)
	print(f"Loaded user embeddings for users, shape: {rlmEm.shape}")
# random 1000 user
all_user_ids = list(user_interactions.keys())


# load item profile at fullMeta_book.csv: iid,asin,title,description,profile
item_profile_path = f'./data/{dataset}/fullMeta_{dataset}.csv'
item_profile = pd.read_csv(item_profile_path)
item_profile = pd.DataFrame(item_profile)

# # load user from sample_profile.json

# user_profile_path = f'./data/{dataset}/sample_user_profile.json'
# if os.path.exists(user_profile_path):
#     print(f"{user_profile_path} exists, load user profile.")
#     with open(user_profile_path, 'r') as f:
#         user_profile = json.load(f)


file_path = f'./data/{dataset}/usr_prf.pkl'
with open(file_path, 'rb') as f:
    user_profile = pickle.load(f)

# from userid get interacted items
user_interactions[0]
selected_user_ids = all_user_ids[:5000]
for user_id in tqdm(selected_user_ids):
    userembed = coldEm[user_id]

    # find top 100 similar users based on cosine similarity between userembed and all other user embeddings
    top100_cold = getTop100SimilarUsers(user_id, coldEm, user_ids, 100)
    top100_rlm = getTop100SimilarUsers(user_id, rlmEm, user_ids, 100)
    
    # embedding 1, get the distance of the farest user in top100 
    farest_distance_cold = top100_cold[-1][1]
    # using embedding of top100 similar, plot the embeddings of top100 similar users in 2D space using TSNE

    top100_user_ids_e1 = [x[0] for x in top100_cold]
    top100_user_ids_e2 = [x[0] for x in top100_rlm]
    matSim1 = [x[0] for x in top100_cold]
    matSim2 = [x[0] for x in top100_rlm]
    
    top100_user_embeddings_e1 = np.stack([coldEm[uid] for uid in top100_user_ids_e1])
    top100_user_embeddings_e2 = np.stack([rlmEm[uid] for uid in top100_user_ids_e2])
    top100_user_embeddings_e1 = np.vstack([top100_user_embeddings_e1, coldEm[user_id]]) # add the user itself
    top100_user_embeddings_e2 = np.vstack([top100_user_embeddings_e2, rlmEm[user_id]]) # add the user itself

    overlap_counts_e1 = countOverlap(user_interactions, user_id, top100_user_ids_e1)
    # get highest 3 overlap users
    top3_overlap_users_e1 = sorted(overlap_counts_e1.items(), key=lambda x: x[1], reverse=True)[:3]
    # if the top 3 user e1 are in top100 similar users of e2, check the distance of these users in e2 to the user.
    for top_user, overlap in top3_overlap_users_e1:
        if top_user in top100_user_ids_e2:
            
            index_e1 = top100_user_ids_e1.index(top_user)
            sim1 = top100_cold[index_e1][1]
            
            index_e2 = top100_user_ids_e2.index(top_user)
            sim2 = top100_rlm[index_e2][1]
            
            if sim1 - sim2 >= 0.2:
                print(sim1, sim2, farest_distance_cold)
                counter = 0
                while True:
                    if counter > 100:
                        break
                    if counter == index_e1:
                         counter += 1
                         continue
                    sim_id = top100_user_ids_e1[counter]
                    try:
                        simid2_e2 = top100_user_ids_e2.index(sim_id)
                        if sim_id != index_e2:
                            break
                    except:
                        counter += 1
                        pass
                if overlap <= 1:
                    continue
                # if (len(user_interactions[user_id]) > 5) or (len(user_interactions[top_user]) > 5):
                #     continue
                print(f"User {user_id} has top overlap user {top_user}, overlapp {overlap} and, detail: {sim1 , sim2 , farest_distance_cold}")
                print(top100_user_ids_e1[counter])
                print("*"*50)
                print("INFO:----------------------------------------------")
                print("*"*50)
                print(user2info(user_id, user_interactions, item_profile))
                print("*"*50)  
                print("INFO of similar user:----------------------------------------------")
                print(user2info(top_user, user_interactions, item_profile))         
                print("*"*50)
                print("INFO of similar user:----------------------------------------------")
                print(user2info(top100_user_ids_e1[counter], user_interactions, item_profile))    
                print("*"*50)
                print("INFO of profile user:----------------------------------------------")
                print(user_profile[user_id])
                print("*"*50)
                print("INFO of similar user:----------------------------------------------")
                print("*"*50)
                print(user_profile[top_user])
                print("*"*50)
                print("INFO of similar user:----------------------------------------------")
                print(user_profile[top100_user_ids_e1[counter]])
                print("END INFO----------------------------------------------")
				