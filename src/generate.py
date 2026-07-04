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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def clean_summary(text):
    text = text.strip()

    # Nếu Qwen vẫn sinh thinking block thì cắt bỏ
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    # Bỏ markdown json wrapper nếu có
    text = text.replace("```json", "").replace("```", "").strip()

    return text


def is_bad_summary(text):
    text = text.strip()

    if len(text) < 5:
        return True

    # Lỗi kiểu .*.*.*.*.*
    if text.count(".*") > 10:
        return True

    # Lỗi sinh nhiều ký tự non-English bất thường
    non_ascii_ratio = sum(ord(c) > 127 for c in text) / max(len(text), 1)
    if non_ascii_ratio > 0.4:
        return True

    # Chuỗi dài nhưng gần như không có space, thường là lặp token
    if len(text) > 300 and text.count(" ") < 10:
        return True

    return False


@torch.inference_mode()
def generate_summary_batch(
    model,
    tokenizer,
    system_prompt,
    batch_contents,
    max_new_tokens=1024,
    do_sample=False,
):
    batch_inputs_text = []

    for content in batch_contents:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        batch_inputs_text.append(input_text)

    # RẤT QUAN TRỌNG với Qwen/Llama/Gemma khi batch generation
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        batch_inputs_text,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    ).to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.5,
        top_p=0.95,
        top_k=20,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    # Với left-padding, toàn bộ input batch có cùng chiều dài.
    # Generate output = padded_input + generated_tokens.
    prompt_len = inputs["input_ids"].shape[-1]
    generated_tokens = output[:, prompt_len:]

    summaries = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    summaries = [clean_summary(s) for s in summaries]
    return summaries


@torch.inference_mode()
def generate_summary_single(
    model,
    tokenizer,
    system_prompt,
    content,
    max_new_tokens=512,
):
    """
    Dùng để retry khi một sample trong batch bị lỗi.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        add_special_tokens=True,
    ).to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.5,
        top_p=0.95,
        top_k=20,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    summary = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return clean_summary(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", "-d", type=str, default="book", help="name of datasets")
    parser.add_argument("--tuning", "-t", type=str2bool, default=False, help="load tuned model or pretrain")
    parser.add_argument("--LLM", type=str, default="06B", help="name of LLM to use: 06B, 4B, or 8B")

    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--out", type=str, default=None)

    parser.add_argument("--prompt_profile", "-pp", type=str2bool, default=True)
    parser.add_argument("--prompt_candidate", "-pc", type=str2bool, default=True)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=3407)

    args, _ = parser.parse_known_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # =========================
    # Load meta data
    # =========================
    meta_path = f"./data/{args.dataset}/fullMeta_{args.dataset}.csv"
    metaDF = pd.read_csv(meta_path)
    metaDF = pd.DataFrame(metaDF)

    unique_meta_asin = set(metaDF["asin"])
    print(f"[Meta] Unique ASINs: {len(unique_meta_asin)}")

    inter_path = f"./data/{args.dataset}/{args.dataset}.inter"
    interDF = pd.read_csv(
        inter_path,
        sep="\t",
        usecols=["userID", "itemID", "x_label"],
    )

    interDF["userID"] = interDF["userID"].astype(int)
    interDF["itemID"] = interDF["itemID"].astype(int)

    # =========================
    # Preparing for users
    # =========================
    user_interactions = getUser_Interaction(interDF)

    # =========================
    # Profiling prompt
    # =========================
    with open("src/prompts.yaml", "r", encoding="utf-8") as f:
        all_prompts = yaml.safe_load(f)

    sys_prompt = all_prompts[args.dataset]["user"]
    itemDesc = get_itemDesc(metaDF)

    # =========================
    # Load item-item KNN nếu vẫn cần check file
    # =========================
    top_k = 10
    item_item_path = f"./data/{args.dataset}/item_top{top_k}item.npy"

    if os.path.exists(item_item_path):
        print(f"{item_item_path} exists, skip building item-item knn.")
        item_kitem = np.load(item_item_path)
    else:
        raise ValueError(
            f"{item_item_path} does not exist, please run preprocess.py to build it."
        )

    # =========================
    # Select model
    # =========================
    if args.tuning:
        selected_model = (
            f"./qwen{args.LLM}_it_model_{args.dataset}"
            f"_candidate_{args.prompt_candidate}"
            f"_profile_{args.prompt_profile}"
        )
    else:
        if args.LLM == "06B":
            selected_model = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
        elif args.LLM == "8B":
            selected_model = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
        elif args.LLM == "4B":
            selected_model = "unsloth/Qwen3-4B-Instruct-2507"
        else:
            raise ValueError(f"Unknown LLM: {args.LLM}")

    print(f"[Model] {selected_model}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=selected_model,
        max_seq_length=4096,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        device_map="balanced",
    )

    FastLanguageModel.for_inference(model)
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =========================
    # Output path
    # =========================
    if args.out is not None:
        user_profile_path = args.out
    else:
        if args.tuning:
            user_profile_path = (
                f"./data/{args.dataset}/tuning{args.LLM}_usr_prf_{args.shard}"
                f"_candidate_{args.prompt_candidate}"
                f"_profile_{args.prompt_profile}.json"
            )
        else:
            user_profile_path = (
                f"./data/{args.dataset}/{args.LLM}_usr_prf_{args.shard}"
                f"_candidate_{args.prompt_candidate}"
                f"_profile_{args.prompt_profile}.json"
            )

    user_profiles = {}

    if os.path.exists(user_profile_path):
        with open(user_profile_path, "r", encoding="utf-8") as f:
            user_profiles = json.load(f)

        print(
            f"Loaded existing user profiles from {user_profile_path}, "
            f"current size: {len(user_profiles)}"
        )

    # =========================
    # Prepare user list
    # =========================
    listUser = list(user_interactions.keys())
    users = listUser[args.shard::args.num_shards]

    pending_uids = []
    pending_contents = []

    # dùng rng riêng để shuffle reproducible
    rng = random.Random(args.seed + args.shard)

    # =========================
    # Batch generation
    # =========================
    for uid in tqdm(users):
        if str(uid) in user_profiles:
            continue

        u_items = list(user_interactions[uid])
        rng.shuffle(u_items)

        itemInfo = "The user has purchased: \n"

        for item in u_items[-10:]:
            itemInfo += str(itemDesc[item][0]).strip() + "\n"

        pending_uids.append(str(uid))
        pending_contents.append(itemInfo)

        if len(pending_contents) >= args.batch_size:
            summaries = generate_summary_batch(
                model=model,
                tokenizer=tokenizer,
                system_prompt=sys_prompt,
                batch_contents=pending_contents,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

            # Retry các output lỗi bằng single generation
            for i, summary in enumerate(summaries):
                if is_bad_summary(summary):
                    print(f"[Retry] Bad summary for user {pending_uids[i]}")
                    summary = generate_summary_single(
                        model=model,
                        tokenizer=tokenizer,
                        system_prompt=sys_prompt,
                        content=pending_contents[i],
                        max_new_tokens=512,
                    )

                user_profiles[pending_uids[i]] = {"summary": summary}

            pending_uids = []
            pending_contents = []

            if len(user_profiles) % 50 == 0:
                with open(user_profile_path, "w", encoding="utf-8") as f:
                    json.dump(user_profiles, f, ensure_ascii=False, indent=4)

    # =========================
    # Process remaining users
    # =========================
    if len(pending_contents) > 0:
        summaries = generate_summary_batch(
            model=model,
            tokenizer=tokenizer,
            system_prompt=sys_prompt,
            batch_contents=pending_contents,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

        for i, summary in enumerate(summaries):
            if is_bad_summary(summary):
                print(f"[Retry] Bad summary for user {pending_uids[i]}")
                summary = generate_summary_single(
                    model=model,
                    tokenizer=tokenizer,
                    system_prompt=sys_prompt,
                    content=pending_contents[i],
                    max_new_tokens=512,
                )

            user_profiles[pending_uids[i]] = {"summary": summary}

    # =========================
    # Final save
    # =========================
    with open(user_profile_path, "w", encoding="utf-8") as f:
        json.dump(user_profiles, f, ensure_ascii=False, indent=4)

    print(f"[Done] Saved {len(user_profiles)} user profiles to {user_profile_path}")