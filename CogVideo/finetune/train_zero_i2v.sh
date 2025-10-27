#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "/project/sds-rise/ethan/explore_diffusion_spatial/huggingface/models/CogVideoX-2b-32in-16out"
    --model_name "cogvideox-i2v"  # ["cogvideox-i2v"]
    --model_type "i2v"
    --training_type "sft"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "/project/sds-rise/ethan/explore_diffusion_spatial/model_finetuned/raw/CogVideoX-2b-32in-16out-7by7"
    --report_to "wandb"
)

# Data Configuration
DATA_ARGS=(
    --data_root "/absolute/path/to/your/data_root"
    --caption_column "prompt.txt"
    --video_column "videos.txt"
    # --image_column "images.txt"  # comment this line will use first frame of video as image conditioning
    --train_resolution "49x480x720"  # (frames x height x width), frames should be 8N+1 and height, width should be multiples of 16
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 10 # number of training epochs
    --seed 42 # random seed

    #########   Please keep consistent with deepspeed config file ##########
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "fp16"  # ["no", "fp16"] Only CogVideoX-2B supports fp16 training
    ########################################################################
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 0
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 5000 # save checkpoint every x steps
    --checkpointing_limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_dir "/absolute/path/to/validation_set"
    --validation_steps 5000  # should be multiple of checkpointing_steps
    --validation_prompts "prompts.txt"
    --validation_images "images.txt"
    --gen_fps 16
)

# Combine all arguments and launch training
/home/jqf8qm/miniforge3/envs/exdif/bin/accelerate launch --config_file /project/sds-rise/ethan/explore_diffusion_spatial/CogVideo/finetune/accelerate_config.yaml /project/sds-rise/ethan/explore_diffusion_spatial/CogVideo/finetune/train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
