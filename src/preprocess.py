import os
import random
import requests
from math import nan
import json
import pandas as pd
import csv
import ast
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import yaml
import gzip
from datasets import Dataset
from helper import build_item_item_knn, get_itemDesc, get_profile_embeddings, getUser_Interaction, get_profile_text

def overlap_items(list1, list2):
	return len(set(list1) & set(list2))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', '-d', type=str, default='movie', help='name of datasets')
	parser.add_argument("--user", '-u', type=bool, default=False, help='encoding user profile or not')
	parser.add_argument("--tuning", '-t', type=bool, default=False, help='using tuned LLM to encode user profile or not')
	parser.add_argument("--item_profile",'-i', type=bool, default=True, help='whether to use item profile or not')
	parser.add_argument("--export_tuning",'-e', type=bool, default=True, help='whether to export tuning data or not')
	parser.add_argument("--export_item", '-ei', type=bool, default=False, help='whether to export item data or not')
	parser.add_argument('--prompt_profile', '-pp', type=bool, default=True, help='ablation: item profile in prompt or not')
	parser.add_argument('--prompt_candidate', '-pc', type=bool, default=True, help='use candidate prompt or not')
	parser.add_argument('--num_neg', '-n', type=int, default=3, help='number of negative samples for tuning')
	args, _ = parser.parse_known_args()
	print(args)
	if args.user:
		dir = f'./data/{args.dataset}/'
		# =========================
		# Load user data
		# =========================
		if args.tuning:
			file_path = f'./data/{args.dataset}/tun_user_profile.json'
		else:
			file_path = f'./data/{args.dataset}/sample_user_profile.json'
		
		with open(file_path, 'r', encoding='utf-8') as jsonfile:
			jsonfile = json.load(jsonfile)

		prf_text = get_profile_text(jsonfile)

		# encode user profiles to embeddings and save as .npy
		user_embeddings = get_profile_embeddings(prf_text, save= True, path = os.path.join(dir, f'user_feat.npy'))
		print("User profile embeddings shape:", user_embeddings.shape)
	else:
		dir = f'./data/{args.dataset}/'
		# =========================
		# Load meta data
		# meta contains information of item
		# iid_asin contains mapping id prf and id meta
		# =========================
		
		# =========================
		# Load train data
		# =========================
		meta_data = []
		file_path = f'./data/{args.dataset}/{args.dataset}.inter'
		interDF = pd.read_csv(file_path, sep="\t", usecols=['userID', 'itemID', 'x_label'])
		interDF['userID'] = interDF['userID'].astype(int)
		interDF['itemID'] = interDF['itemID'].astype(int)
		# metaDF = pd.DataFrame(interDF)
		num_users = interDF['userID'].nunique()
		print(num_users)

		users_by_label = {
		l: set(interDF[interDF['x_label'] == l]['userID'].unique())
		for l in [0, 1, 2]
		}

		print("0 ∩ 1:", len(users_by_label[0] & users_by_label[1]))
		print("1 ∩ 2:", len(users_by_label[1] & users_by_label[2]))
		print("0 ∩ 2:", len(users_by_label[0] & users_by_label[2]))
		
		if args.dataset in ['book', 'movie']:
			meta_csv_path = os.path.join(dir, f'meta_{args.dataset}.json.gz')
		else:
			meta_csv_path = os.path.join(dir, f'yelp_academic_dataset_business.json')
		
		# if dataset in ['book', 'yelp']: load iid_asin else skip
		
		if args.dataset in ['book', 'yelp']:
			iid_asin_path = os.path.join(dir, f"{args.dataset}_asin.json")
			iid_asin = {}
			# -------- read JSON Lines --------
			records = []
			with open(iid_asin_path, "r", encoding="utf-8") as f:
				for line in f:
					line = line.strip()
					if line:
						records.append(json.loads(line))

			iid_df = pd.DataFrame(records)   # columns: iid, asin
			# rename business_id to asin for yelp
			if args.dataset == 'yelp':
				iid_df = iid_df.rename(columns={'business_id': 'asin'})
			iid_asin_set = set(iid_df['asin'].tolist())
			# print(iid_df.head())
			print(f"Number of items in iid_asin: {len(iid_asin_set)}")
			print(f"Sample iid_asin: {iid_df.sample(5)}")


		


			# =========================
			# Load item data
			# =========================
			file_path = f'./data/{args.dataset}/itm_prf.pkl'
			with open(file_path, 'rb') as f:
				prf = pickle.load(f)
			
			# check all items in metaDF appear in prf
			meta_items = set(interDF['itemID'].unique())
			prf_items = set(prf.keys())
			print("Number of items in metaDF and prf:", len(meta_items & prf_items))
			print("Number of items only in metaDF:", len(meta_items - prf_items))

			prf_text = []
			for idx in prf_items:
				item_profile = prf[idx]['profile']
				prf_text.append(item_profile)

			

			# random a single sample of item profiles
			randomID = random.choice(list(prf_items))
			print("An item profile contains:", prf[randomID].keys(), "sample item:", prf[randomID])

			metaDF_filtered_path = os.path.join(dir, f'metaDF_filtered_{args.dataset}.csv')
			if os.path.exists(metaDF_filtered_path) is False:
				# load, clean, using the selected data
				data = []
				if args.dataset in ['book', 'movie']:
					with gzip.open(meta_csv_path, 'rt') as f:
						for line in tqdm(f):
							tmp = ast.literal_eval(line)
							if tmp['asin'] not in iid_asin_set:
								continue
							data.append(tmp)

					metaDF = pd.DataFrame(data)
					metaDF_filtered = metaDF[["asin", "title", "description"]].copy()
				else:
					with open(meta_csv_path, 'r', encoding='utf-8') as f:
						for line in tqdm(f):
							tmp = json.loads(line)
							if tmp['business_id'] not in iid_asin_set:
								continue
							data.append(tmp)

					metaDF = pd.DataFrame(data)
					metaDF_filtered = metaDF[["business_id", "name", "address", "city"]].copy()
					# combine address and city as description
					metaDF_filtered['categories'] = metaDF_filtered['address'] + ', ' + metaDF_filtered['city']
					# rename columns
					metaDF_filtered = metaDF_filtered.rename(columns={
						'business_id': 'asin',
						'name': 'title',
						'categories': 'description'
					})
				# saving metaDF_filtered
				metaDF_filtered.to_csv(metaDF_filtered_path, index=False)
			else:
				metaDF_filtered = pd.read_csv(metaDF_filtered_path)
			# interDF: id, asin
			# metaDF_filtered: asin, title, description
			# merge interDF with metaDF_filtered
			# 

			print("interDF columns:", iid_df.columns.tolist())
			print(iid_df.head())

			print("\nmetaDF_filtered columns:", metaDF_filtered.columns.tolist())
			print(metaDF_filtered.head())


			merged_df = iid_df.merge(
				metaDF_filtered[["asin", "title", "description"]],
				on="asin",
				how="left"
			)
			merged_df = merged_df.sort_values(by='iid').reset_index(drop=True)
		
		
			# print number of missing both titles and descriptions
			num_missing_both = merged_df['title'].isnull() & merged_df['description'].isnull()
			print(f"Number of missing both titles and descriptions: {num_missing_both.sum()}")
			# print number of rows in merged_df
			print(f"Number of rows in merged_df: {merged_df.shape[0]}")

			# fill null titles or descriptions with empty string
			merged_df['title'] = merged_df['title'].fillna('')
			merged_df['description'] = merged_df['description'].fillna('')
			merged_df['profile'] = prf_text

			# create new column with combine title and description
			merged_df['text_feat'] = merged_df['title'] + ' ' + merged_df['profile']
		else:
			# load meta data directly at item_meta.csv
			metapath = os.path.join(dir, f'fullMeta_movie.csv')
			merged_df = pd.read_csv(metapath)

		if args.export_item or (not os.path.exists(os.path.join(dir, f'text_feat.npy'))):
			if args.item_profile:
				# save prf embeddings as .npy in the order of itemID
				text_embeddings = get_profile_embeddings(merged_df['text_feat'].tolist(), path = os.path.join(dir, f'text_feat.npy'))
			else:	
				# encode text_feat to embeddings and save as .npy
				merged_df['text_feat'] = merged_df['title'] + ' ' + merged_df['description']
				text_embeddings = get_profile_embeddings(merged_df['text_feat'].tolist(), path = os.path.join(dir, f'text_feat.npy'))
		else:
			# load existing text_feat embeddings
			text_embeddings = np.load(os.path.join(dir, f'text_feat.npy'))
		top_k = 10
		item_kitem = build_item_item_knn(text_embeddings, top_k=top_k)
		item_item_path = f'./data/{args.dataset}/item_top{top_k}item.npy'
		np.save(item_item_path, item_kitem)

		user_interactions = getUser_Interaction(interDF)
		itemDesc = get_itemDesc(merged_df, merge=False)
		checkarray = []
		listUser = list(user_interactions.keys())
		if args.export_tuning:
			with open("src/prompts.yaml", "r") as f:
				all_prompts = yaml.safe_load(f)
			
			tuningP, systemP1, systemP2 = "tuning", "sys", "user"
			if args.prompt_candidate:
				tuningP, systemP1, systemP2 = "tuning_candidate", "sys_candidate", "user"
			tun_prompt1 = all_prompts[args.dataset]["tuning"]
			tun_prompt2 = all_prompts[args.dataset]["tuning_candidate"]
			sys_prompt1 = all_prompts[args.dataset][systemP1]
			sys_prompt2 = all_prompts[args.dataset][systemP2]

			with open(f"./data/{args.dataset}/sample_user_profile.json", 'r', encoding='utf-8') as f:
				sampleUser = json.load(f)

			tuningLLM_name = 'QwenTuning'
			dataset = []
			for uid in tqdm(listUser):
				u_items = user_interactions[uid]
				selected = u_items[-10:] 
				if len(dataset) % 2 == 0:
					ground_truth = selected[-1]
					interacted = selected[:-1]
					itemInfo = ""
					for item in interacted:
						title, description = itemDesc[item]
						choose_def = ""
						if args.prompt_profile is True:
							choose_def = f"Description: {description}\n"
						tmp = f"Title: {title}\n{choose_def}\n"
						itemInfo += tmp

					candidates = item_kitem[ground_truth]
					listC = []
					for c in candidates:
						if c in u_items:
							continue
						listC.append(c)
					random.shuffle(listC)
					listC = listC[:args.num_neg]
					checkarray.append(len(listC))
					candidateInfo = ""
					# must add ground_truth to candidates
					listC.append(ground_truth)
					random.shuffle(listC)
					for c in listC:
						title, description = itemDesc[c]
						choose_def = ""
						if args.prompt_profile is True:
							choose_def = f"Description: {description}\n"
						tmp = f"Title: {title}\n{choose_def}\n"
						candidateInfo += tmp
					if args.prompt_candidate:
						userprompt = tun_prompt2.format(itemInfo, candidateInfo)
					else:
						userprompt = tun_prompt1.format(itemInfo, "")
					answer = f"{itemDesc[ground_truth][0]}"
					if answer == "":
						continue
					sys_prompt = sys_prompt1
				else:
					itemInfo = ""
					for item in selected:
						title, description = itemDesc[item]
						choose_def = ""
						if args.prompt_profile is True:
							choose_def = f"Description: {description}\n"
						tmp = f"Title: {title}\n{choose_def}\n"
						itemInfo += tmp
					userprompt = tun_prompt1.format(itemInfo, "")
					sys_prompt = sys_prompt2
					answer = str(sampleUser[str(uid)])


				dataset.append({
					"userprompt": userprompt,
					"systemprompt": sys_prompt,
					"answer": answer
				})

			
			dataset = Dataset.from_list(dataset)
			dataset.to_json(f"./data/{args.dataset}/candidate_{args.prompt_candidate}_profile_{args.prompt_profile}_tuningData_{args.num_neg}.jsonl")
			# stat for candidate
			print(np.mean(checkarray), np.min(checkarray), np.max(checkarray))	