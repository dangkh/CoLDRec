from helper import get_profile_embeddings, get_profile_text
import os
import json
import ast

dataset = 'book'
dir = f'./data/{dataset}/'
# =========================
# Load user data
# =========================
file_path = f'./data/{dataset}/tuning_user_profile.json'

with open(file_path, 'r', encoding='utf-8') as jsonfile:
	jsonfile = json.load(jsonfile)


counter = 0
counter2 = 0
counter3 = 0
profile = []
for ii in range(len(jsonfile)):
	try:
		level1 = ast.literal_eval(jsonfile[str(ii)]['summary'])
		summarization = level1["summarization"]
	except Exception:
		try:
			level1 = ast.literal_eval(jsonfile[str(ii)]['summary'])
			level2 = ast.literal_eval(level1['summary'])
			summarization = level2['summarization']
			counter2 += 1
		except Exception:
			try:
				summarization = jsonfile[str(ii)]['summary']
				counter3 += 1
			except Exception:
				print(f"Error parsing JSON for key: {ii}")
				print(f"Value: {jsonfile[str(ii)]}")
	profile.append(summarization)
	counter += 1
print(f"Total successfully parsed profiles: {counter} , {counter2} with extra level, {counter3} with direct text")
print(f"Total profiles parsed with encoding errors: {counter2}")

# encode user profiles to embeddings and save as .npy
user_embeddings = get_profile_embeddings(profile, save = True, path = os.path.join(dir, f'new_user_feat.npy'))
print("User profile embeddings shape:", user_embeddings.shape)