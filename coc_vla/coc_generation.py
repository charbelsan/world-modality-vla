from __future__ import annotations

import argparse
import json
import os
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Chain-of-Causality (CoC) text for episodes.")
    parser.add_argument("--dataset_name", type=str, required=True, help="LeRobot dataset name (e.g. HuggingFaceVLA/libero).")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--image_key", type=str, default="rgb")
    parser.add_argument("--instruction_key", type=str, default="instruction", help="Key for instruction text, if available.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Where to write episode-level CoC JSONL.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Open-source VLM model name.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_episodes", type=int, default=1000, help="Max episodes to annotate (for prototyping).")
    return parser.parse_args()


def load_vlm(model_name: str, device: str):
    """
    Load an open-source VLM via transformers.
    This is a sketch; you can adapt it to the actual model you use.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()
    return tokenizer, model


def generate_coc_for_episode(
    tokenizer,
    model,
    device: str,
    instruction: str,
    frames: List[Image.Image],
    max_new_tokens: int = 256,
) -> str:
    """
    Generate a chain-of-causality text for a single episode.

    NOTE: This is a placeholder that uses text-only prompting. In practice you
    should adapt it to pass images to the chosen VLM (e.g. Qwen2-VL, LLaVA).
    """
    # Simple text-only prompt for now.
    prompt = (
        "You see a robot executing a task.\n"
        f"Task instruction: {instruction}\n"
        "Based on the start, middle, and end of the episode, describe the chain of causality "
        "of the robot's behavior in 3-6 short numbered steps.\n"
        "Each step should say what the robot does and why that helps achieve the goal.\n"
        "Format:\n"
        "1. ...\n"
        "2. ...\n"
        "3. ...\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Heuristic: return the part after the prompt.
    return text[len(prompt) :].strip()


def main():
    args = parse_args()

    ds = LeRobotDataset(args.dataset_name)  # load full dataset

    tokenizer, model = load_vlm(args.model_name, args.device)

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    num_episodes = ds.num_episodes if hasattr(ds, "num_episodes") else 1

    with open(args.output_jsonl, "w") as out_f:
        count = 0
        for ep_id in tqdm(range(num_episodes), desc="Generating CoC"):
            if count >= args.max_episodes:
                break

            # Simple approach: assume dataset has get_episode method.
            if hasattr(ds, "get_episode"):
                episode = ds.get_episode(ep_id)
            else:
                # Fallback: treat entire dataset as one long episode
                episode = [ds[i] for i in range(len(ds))]

            if len(episode) == 0:
                continue

            # Instruction text.
            step0 = episode[0]
            instruction = step0.get(args.instruction_key, "")

            # Frames: pick start, mid, end indices (not passed to the VLM in this sketch).
            t0 = 0
            tm = len(episode) // 2
            te = len(episode) - 1

            frames = []
            for t in [t0, tm, te]:
                step = episode[t]
                img = step[args.image_key]  # [C,H,W] or PIL
                if isinstance(img, torch.Tensor):
                    img_np = img.permute(1, 2, 0).cpu().numpy()
                    frames.append(Image.fromarray(img_np))
                elif isinstance(img, np.ndarray):
                    frames.append(Image.fromarray(img))
                elif isinstance(img, Image.Image):
                    frames.append(img)

            coc_text = generate_coc_for_episode(
                tokenizer=tokenizer,
                model=model,
                device=args.device,
                instruction=instruction,
                frames=frames,
            )

            obj = {
                "episode_id": ep_id,
                "instruction": instruction,
                "coc_text": coc_text,
            }
            out_f.write(json.dumps(obj) + "\n")
            count += 1


if __name__ == "__main__":
    main()

