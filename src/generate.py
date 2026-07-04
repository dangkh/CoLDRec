import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from unsloth import FastLanguageModel

from helper import get_itemDesc, getUser_Interaction


def get_message(system_prompt, content):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def strip_qwen_thinking(text: str) -> str:
    """
    Safety cleanup in case Qwen still emits <think>...</think>.
    """
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()


@torch.inference_mode()
def generate_summary(
    model,
    tokenizer,
    batch_messages,
    max_new_tokens=512,
    enable_thinking=False,
):
    all_prompts = []

    for messages in batch_messages:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        all_prompts.append(input_text)

    inputs = tokenizer(
        all_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
        add_special_tokens=False,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }

    if enable_thinking:
        # Qwen3 docs recommend sampling for thinking mode.
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
            }
        )
    else:
        # For profile generation, deterministic output is usually better.
        generation_kwargs.update(
            {
                "do_sample": False,
            }
        )

    outputs = model.generate(
        **inputs,
        **generation_kwargs,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], outputs)
    ]

    output_texts = tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return [strip_qwen_thinking(x) for x in output_texts]


def save_json_atomic(data, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", "-d", type=str, default="book")
    parser.add_argument("--tuning", "-t", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        help="Base Qwen 0.6B model from Unsloth.",
    )
    parser.add_argument(
        "--tuned_model_path",
        type=str,
        default=None,
        help="Path to your tuned Qwen 0.6B model. Used when --tuning is enabled.",
    )

    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--out", type=str, default=None)

    parser.add_argument("--prompt_profile", "-pp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt_candidate", "-pc", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=False)

    args, _ = parser.parse_known_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # =========================
    # Load metadata
    # =========================
    meta_path = f"./data/{args.dataset}/fullMeta_{args.dataset}.csv"
    inter_path = f"./data/{args.dataset}/{args.dataset}.inter"

    metaDF = pd.read_csv(meta_path)
    unique_meta_asin = set(metaDF["asin"])
    print(f"[Meta] Unique ASINs: {len(unique_meta_asin)}")

    interDF = pd.read_csv(
        inter_path,
        sep="\t",
        usecols=["userID", "itemID", "x_label"],
    )
    interDF["userID"] = interDF["userID"].astype(int)
    interDF["itemID"] = interDF["itemID"].astype(int)

    # =========================
    # Prepare users
    # =========================
    user_interactions = getUser_Interaction(interDF)

    # =========================
    # Load prompts
    # =========================
    with open("src/prompts.yaml", "r", encoding="utf-8") as f:
        all_prompts = yaml.safe_load(f)

    sys_prompt = all_prompts[args.dataset]["user"]
    itemDesc = get_itemDesc(metaDF)

    # =========================
    # Load Qwen 0.6B
    # =========================
    if args.tuning:
        selected_model = args.tuned_model_path
        if selected_model is None:
            selected_model = (
                f"./qwen0_6B_it_model_{args.dataset}"
                f"_candidate_{args.prompt_candidate}"
                f"_profile_{args.prompt_profile}"
            )
    else:
        selected_model = args.model_name

    print(f"[Model] Loading: {selected_model}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=selected_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        device_map="auto",
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
        user_profile_path = (
            f"./data/{args.dataset}/usr_prf_qwen0_6B_{args.shard}"
            f"_candidate_{args.prompt_candidate}"
            f"_profile_{args.prompt_profile}.json"
        )

    user_profiles = {}
    if os.path.exists(user_profile_path):
        with open(user_profile_path, "r", encoding="utf-8") as f:
            user_profiles = json.load(f)
        print(
            f"[Resume] Loaded existing user profiles from {user_profile_path}, "
            f"current size: {len(user_profiles)}"
        )

    # =========================
    # Build batches
    # =========================
    list_users = list(user_interactions.keys())
    users = list_users[args.shard::args.num_shards]

    batch_messages = []
    q_ids = []
    q_messages = []

    rng = random.Random(args.seed + args.shard)

    for uid in tqdm(users, desc="Building prompts"):
        if str(uid) in user_profiles:
            continue

        u_items = list(user_interactions[uid])
        rng.shuffle(u_items)

        item_info = "The user has purchased the following items:\n"

        for item in u_items[-10:]:
            if item in itemDesc:
                item_info += str(itemDesc[item]).strip() + "\n"

        messages = get_message(sys_prompt, item_info)
        q_ids.append(str(uid))
        q_messages.append(messages)

        if len(q_messages) >= args.batch_size:
            batch_messages.append((q_ids, q_messages))
            q_ids = []
            q_messages = []

    if len(q_messages) > 0:
        batch_messages.append((q_ids, q_messages))

    print(f"[Batch] Number of batches: {len(batch_messages)}")

    # =========================
    # Save first batch for debugging
    # =========================
    if len(batch_messages) > 0:
        debug_path = (
            f"./data/{args.dataset}/batch_messages_qwen0_6B_{args.shard}"
            f"_candidate_{args.prompt_candidate}"
            f"_profile_{args.prompt_profile}.txt"
        )

        with open(debug_path, "w", encoding="utf-8") as f:
            first_ids, first_messages = batch_messages[0]

            for uid, messages in zip(first_ids, first_messages):
                f.write(f"User ID: {uid}\n")
                for msg in messages:
                    f.write(f"{msg['role']}: {msg['content']}\n")
                f.write("\n====================\n\n")

        print(f"[Debug] Saved first batch to {debug_path}")

    # =========================
    # Generate profiles
    # =========================
    for batch_idx, (batch_ids, batch_info) in enumerate(
        tqdm(batch_messages, desc="Generating profiles")
    ):
        summaries = generate_summary(
            model=model,
            tokenizer=tokenizer,
            batch_messages=batch_info,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
        )

        for uid, summary in zip(batch_ids, summaries):
            user_profiles[str(uid)] = {"summary": summary}

        if (batch_idx + 1) % 10 == 0:
            save_json_atomic(user_profiles, user_profile_path)
            print(f"[Save] Saved {len(user_profiles)} profiles to {user_profile_path}")

    save_json_atomic(user_profiles, user_profile_path)
    print(f"[Done] Saved {len(user_profiles)} profiles to {user_profile_path}")


if __name__ == "__main__":
    main()