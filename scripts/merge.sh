python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /path/to/your/checkpoints/global_step_190/actor \
    --target_dir /path/to/your/models/merged_model