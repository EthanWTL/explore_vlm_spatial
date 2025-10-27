# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Sample GenEval images."""

import argparse
import os
import json

import torch
import PIL
import tqdm

from diffnext.pipelines.builder import build_pipeline, get_pipeline_path


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="sample geneval images")
    parser.add_argument("--metadata", type=str, help="JSONL metadata")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint file")
    parser.add_argument("--prompt", type=str, default="", help="prompt pth file")
    parser.add_argument("--num_pred_steps", type=int, default=128, help="inference steps")
    parser.add_argument("--num_diff_steps", type=int, default=25, help="diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7, help="guidance scale")
    parser.add_argument("--prompt_size", type=int, default=16, help="prompt size for each batch")
    parser.add_argument("--sample_size", type=int, default=4, help="sample size for each prompt")
    parser.add_argument("--vae_batch_size", type=int, default=16, help="vae batch size")
    parser.add_argument("--outdir", type=str, default="", help="write to")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prompt_dict = torch.load(args.prompt, weights_only=False)

    num_pred_steps, num_diff_steps = args.num_pred_steps, args.num_diff_steps
    gen_args = {"num_inference_steps": num_pred_steps, "num_diffusion_steps": num_diff_steps}
    img_args = {"guidance_scale": args.guidance_scale, "output_type": "np"}
    img_args["vae_batch_size"] = args.vae_batch_size

    rank, world_size = 0, 1
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device), torch.manual_seed(1337)
    generator = torch.Generator(device).manual_seed(1337)
    gen_args.update({"generator": generator, "disable_progress_bar": True})
    is_root = device.index == 0

    pipe_path = get_pipeline_path(args.ckpt, {"text_encoder": ""})
    pipe = build_pipeline(pipe_path, "nova", precison="float16").to(device=device)

    prompts = prompt_dict["prompts"]
    metadatas = [json.loads(v) for v in open(args.metadata)]
    os.makedirs(args.outdir, exist_ok=True) if is_root else None

    grids, prompt_inds = (args.prompt_size, args.sample_size), []
    rank_prompt_inds = list(range(len(prompts)))[slice(rank, None, world_size)]

    for i, idx in enumerate(tqdm.tqdm(rank_prompt_inds, disable=not is_root)):
        prompt_inds.append(idx)
        if len(prompt_inds) != grids[0] and i != len(rank_prompt_inds) - 1:
            continue
        batch_prompts = sum([[prompts[i]] * grids[1] for i in prompt_inds], [])
        outputs = pipe(prompt_embeds=batch_prompts, **img_args, **gen_args)
        img = outputs["frames"][:, 0] if "frames" in outputs else outputs["images"]
        for i, idx in enumerate(prompt_inds):
            out_path = os.path.join(args.outdir, f"{idx:0>5}")
            sample_path = os.path.join(out_path, "samples")
            os.makedirs(out_path, exist_ok=True), os.makedirs(sample_path, exist_ok=True)
            json.dump(metadatas[idx], open(os.path.join(out_path, "metadata.jsonl"), "w"))
            for j in range(grids[1]):
                pil_img = PIL.Image.fromarray(img[i * grids[1] + j])
                pil_img.save(os.path.join(sample_path, f"{j:05}.png"))
        prompt_inds = []
