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


dataset = "yelp"
file_path = f'./data/{dataset}/{dataset}.inter'
interDF = pd.read_csv(file_path, sep="\t", usecols=['userID', 'itemID', 'x_label'])
interDF['userID'] = interDF['userID'].astype(int)
interDF['itemID'] = interDF['itemID'].astype(int)

user_interactions = getUser_Interaction(interDF)

# get user has less than 3 interactions
cold_users = []
for uid, interactions in user_interactions.items():
	if len(interactions) <= 5:
		cold_users.append(uid)

print(f"Number of cold users: {len(cold_users)}")
# save cold users to file
with open(f'./data/{dataset}/cold_users.pkl', 'wb') as f:
	pickle.dump(cold_users, f)