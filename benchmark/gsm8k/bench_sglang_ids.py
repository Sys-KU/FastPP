"""
GSM8K benchmark that bypasses server-side tokenization.

Instead of sending raw `text` (which the server tokenizes serially in its
single asyncio event loop), we tokenize the prompt on the client side and
send `input_ids` directly. This lets all requests land in the scheduler's
waiting queue nearly simultaneously, so you can observe true server-side
concurrency / throughput.

Usage:
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-32B-Instruct --port 30000

    python3 bench_sglang_ids.py \
        --base-url http://127.0.0.1:30000 \
        --tokenizer Qwen/Qwen2.5-32B-Instruct \
        --num-questions 1000 \
        --parallel 1000
"""

import argparse
import ast
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from sglang.utils import download_and_cache_file, read_jsonl

INVALID = -9999999


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def send_one_request(base_url, input_ids, sampling_params, idx):
    """Send a single /generate request with pre-tokenized input_ids.

    The server skips tokenization when input_ids is provided
    (tokenizer_manager.py:240), so the request goes straight to the
    scheduler waiting queue.
    """
    payload = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
    }
    resp = requests.post(base_url + "/generate", json=payload, timeout=600)
    resp.raise_for_status()
    obj = resp.json()
    return idx, obj["text"], obj["meta_info"]


def main(args):
    # Load tokenizer on the client side
    print(f"Loading tokenizer from {args.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )

    # Read data
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    filename = download_and_cache_file(url)
    lines = list(read_jsonl(filename))

    # Construct prompts (same as bench_sglang.py)
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)

    # Pre-tokenize all prompts on the client side
    full_prompts = [few_shot_examples + q for q in questions]
    print(f"Tokenizing {len(full_prompts)} prompts on the client side ...")
    tic = time.time()
    all_input_ids = [
        tokenizer.encode(p, add_special_tokens=False) for p in full_prompts
    ]
    tok_latency = time.time() - tic
    lens = [len(x) for x in all_input_ids]
    print(
        f"Tokenization done in {tok_latency:.3f}s. "
        f"Prompt token lengths: min={min(lens)}, max={max(lens)}, "
        f"mean={np.mean(lens):.1f}, median={np.median(lens):.0f}"
    )

    sampling_params = {
        "temperature": 0,
        "max_new_tokens": args.max_new_tokens,
        "stop": ["Question", "Assistant:", "<|separator|>"],
    }

    # Fire all requests concurrently via a thread pool.
    # Each thread blocks on its HTTP response, but because we send
    # input_ids (no server-side tokenization), all requests enter the
    # scheduler waiting queue almost instantly.
    base_url = args.base_url.rstrip("/")
    results = [None] * len(all_input_ids)

    num_threads = min(args.parallel, len(all_input_ids))
    print(
        f"Sending {len(all_input_ids)} requests concurrently "
        f"with {num_threads} threads ..."
    )
    tic = time.time()

    with ThreadPoolExecutor(num_threads) as executor:
        futures = []
        for i, ids in enumerate(all_input_ids):
            futures.append(
                executor.submit(
                    send_one_request, base_url, ids, sampling_params, i
                )
            )
        for f in tqdm(
            futures, total=len(futures), desc="Waiting for responses"
        ):
            idx, text, meta = f.result()
            results[idx] = (text, meta)

    latency = time.time() - tic

    # Score
    preds = [get_answer_value(text) for text, _ in results]
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    num_output_tokens = sum(meta.get("completion_tokens", 0) for _, meta in results)
    output_throughput = num_output_tokens / latency if latency > 0 else 0.0

    print()
    print(f"Accuracy:         {acc:.3f}")
    print(f"Invalid:          {invalid:.3f}")
    print(f"Latency:          {latency:.3f} s")
    print(f"Output throughput:{output_throughput:.3f} token/s")

    # Dump results
    with open(args.output_file, "w") as fout:
        for i, (text, meta) in enumerate(results):
            fout.write(
                json.dumps(
                    {
                        "index": i,
                        "prompt_tokens": meta.get("prompt_tokens", -1),
                        "completion_tokens": meta.get("completion_tokens", -1),
                        "pred": preds[i],
                        "label": labels[i],
                        "text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Per-request results written to {args.output_file}")

    if args.result_file:
        with open(args.result_file, "a") as fout:
            value = {
                "task": "gsm8k",
                "backend": "srt-ids",
                "latency": round(latency, 3),
                "accuracy": round(acc, 3),
                "num_requests": len(all_input_ids),
                "parallel": args.parallel,
                "throughput_token_per_s": round(output_throughput, 3),
            }
            fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url", type=str, default="http://127.0.0.1:30000"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="HuggingFace model id or local path for the tokenizer "
        "(should match the server model).",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--parallel", type=int, default=1000)
    parser.add_argument("--output-file", type=str, default="tmp_output_ids.jsonl")
    parser.add_argument("--result-file", type=str, default=None)
    args = parser.parse_args()
    main(args)
