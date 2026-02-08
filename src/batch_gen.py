from math import nan
import pandas as pd
from tqdm import tqdm
import yaml
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import random
from unsloth import FastLanguageModel
import torch
import json
from helper import build_item_item_knn, get_itemDesc, getUser_Interaction


def get_message(system_prompt, content): 
	messages = [
		{"role": "system", "content": system_prompt},
		{"role" : "user", "content" : content}
	]
	return messages

def generate_summary(model, tokenizer, batchInfo):
	allMessages = []
	for messages in batchInfo:
		input_text = tokenizer.apply_chat_template(
			messages,
			tokenize = False,
			add_generation_prompt = True, # Must add for generation
			enable_thinking = True
		)
		allMessages.append(input_text)

	inputs = tokenizer(
		allMessages,
		return_tensors = "pt",
		add_special_tokens = True,
		padding=True,
	).to("cuda")

	output = model.generate(
		**inputs,
		max_new_tokens = 1024, # Increase for longer outputs!
		temperature = 0.5, top_p = 0.95, top_k = 20, # For non thinking
		do_sample = False
	)
	generated_ids_trimmed = [
    	out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)
	]
	output_texts = tokenizer.batch_decode(
		generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
	)
	return output_texts

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', '-d', type=str, default='book', help='name of datasets')
	parser.add_argument('--tuning',  '-t', type=bool, default=True, help='load tuned model or pretrain')
	parser.add_argument('--LLM', type=str, default='Llama', help='name of LLM to use: Llama or Gemma, Qwen')
	parser.add_argument("--shard", type=int, default=0)
	parser.add_argument("--num_shards", type=int, default=1)
	parser.add_argument("--out", type=str, default="sample_user_profile.json")
	parser.add_argument('--prompt_profile', '-pp', type=bool, default=True, help='ablation: item profile in prompt or not')
	parser.add_argument('--prompt_candidate', '-pc', type=bool, default=True, help='use candidate prompt or not')
	args, _ = parser.parse_known_args()
	print(args)

	# =========================
	# Load meta data
	# =========================
	meta_data = []
	file_path = f'./data/{args.dataset}/fullMeta_{args.dataset}.csv'
	metaDF = pd.read_csv(file_path)
	metaDF = pd.DataFrame(metaDF)
	unique_meta_asin = set(metaDF['asin'])
	print(f"[Meta] Unique ASINs: {len(unique_meta_asin)}")

	file_path = f'./data/{args.dataset}/{args.dataset}.inter'
	interDF = pd.read_csv(file_path, sep="\t", usecols=['userID', 'itemID', 'x_label'])
	interDF['userID'] = interDF['userID'].astype(int)
	interDF['itemID'] = interDF['itemID'].astype(int)

	# =========================
	# Preparing for users
	# =========================

	user_interactions = getUser_Interaction(interDF)

	# =========================
	# Profiling for user
	# =========================
	with open("src/prompts.yaml", "r") as f:
		all_prompts = yaml.safe_load(f)
	sys_prompt = all_prompts[args.dataset]['user']


	itemDesc = get_itemDesc(metaDF)

	# for each item, find top-k similar items
	top_k = 10
	item_item_path = f'./data/{args.dataset}/item_top{top_k}item.npy'
	if os.path.exists(item_item_path):
		print(f"{item_item_path} exists, skip building item-item knn.")
		item_kitem = np.load(item_item_path)
	else:
		raise ValueError(f"{item_item_path} does not exist, please run preprocess.py to build it.")


	fourbit_models = [
		"unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit", # Qwen 14B 2x faster
		"unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
		"unsloth/Qwen3-8B-unsloth-bnb-4bit",
		"unsloth/Qwen3-14B-unsloth-bnb-4bit",
		"unsloth/Qwen3-32B-unsloth-bnb-4bit",

		# 4bit dynamic quants for superior accuracy and low memory use
		"unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
		"unsloth/Phi-4",
		"unsloth/Llama-3.1-8B",
		"unsloth/Llama-3.2-3B",
		"unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # [NEW] We support TTS models!
	] # More models at https://huggingface.co/unsloth

	selected_model = "unsloth/Qwen3-4B-Instruct-2507"
	if args.tuning:
		selected_model = f"./qwen4B_it_model_{args.dataset}_candidate_{args.prompt_candidate}_profile_{args.prompt_profile}"

	print(selected_model)
	
	model, tokenizer = FastLanguageModel.from_pretrained(
		model_name = selected_model,
		max_seq_length = 4096, # Choose any for long context!
		load_in_4bit = True,  # 4 bit quantization to reduce memory
		load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
		full_finetuning = False, # [NEW!] We have full finetuning now!
		device_map = "balanced",
		# token = "hf_...", # use one if using gated models
	)


	user_profiles = {}
	checkarray = []
	listUser = list(user_interactions.keys())
	users = listUser[args.shard::args.num_shards]

	user_profile_path = f'./data/{args.dataset}/usr_prf_{args.shard}_candidate_{args.prompt_candidate}_profile_{args.prompt_profile}.json'
	if os.path.exists(user_profile_path):
		with open(user_profile_path, 'r', encoding='utf-8') as f:
			user_profiles = json.load(f)
		print(f"Loaded existing user profiles from {user_profile_path}, current size: {len(user_profiles)}")
	q_message = []
	q_id = []
	batch_size = 16
	batch_messages = []
	for uid in tqdm(users):
		if str(uid) in user_profiles:
			continue
		u_items = user_interactions[uid]
		random.shuffle(u_items)
		itemInfo = "The user has purchased: \n"
		for item in u_items[-10:]:
			itemInfo += itemDesc[item]

		messages = get_message(sys_prompt, itemInfo)
		q_message.append(messages)
		q_id.append(str(uid))
		if len(q_message) >= batch_size:
			batch_messages.append([q_id, q_message])
			q_message = []
			q_id = []
	

	if len(q_message) > 0:
		batch_messages.append([q_id, q_message])
		q_message = []
		q_id = []
		
	# save batch_messages[0] to file text for debugging
	with open(f'./data/{args.dataset}/batch_messages_{args.shard}_candidate_{args.prompt_candidate}_profile_{args.prompt_profile}.txt', 'w', encoding='utf-8') as f:
		for uid, messages in zip(batch_messages[0][0], batch_messages[0][1]):
			f.write(f"User ID: {uid}\n")
			for msg in messages:
				f.write(f"{msg['role']}: {msg['content']}\n")
			f.write("\n====================\n\n")
	

	
	for batchId, batchInfo in tqdm(batch_messages):
		summary = generate_summary(model, tokenizer, batchInfo)
		for i, uid in enumerate(batchId):
			user_profiles[str(uid)] = { "summary": summary[i] }

		if (len(user_profiles)) % (batch_size * 10) == 0:
			with open(user_profile_path, 'w', encoding='utf-8') as f:
				json.dump(user_profiles, f, ensure_ascii=False, indent=4)

	with open(user_profile_path, 'w', encoding='utf-8') as f:
		json.dump(user_profiles, f, ensure_ascii=False, indent=4)
	
	# stat for candidate
	# print(np.mean(checkarray), np.min(checkarray), np.max(checkarray))
	


