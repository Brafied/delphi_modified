from pathlib import Path
from transformers import AutoTokenizer
from delphi.__main__ import load_artifacts, populate_cache
from delphi.config import ConstructorConfig, RunConfig, CacheConfig, SamplerConfig

cache_config = CacheConfig(
    dataset_repo="lmsys/lmsys-chat-1m",
    dataset_split="train[:50%]", # MODIFY
    dataset_name="",
    batch_size=325, # MODIFY
    cache_ctx_len=2000,
    n_splits=1,
)
run_config = RunConfig(
    cache_cfg=cache_config,
    constructor_cfg=ConstructorConfig(),
    sampler_cfg=SamplerConfig(),
    model="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    sparse_model="/scratch/general/vast/u1307785/sparsify_checkpoints/Skywork_Skywork-Reward-V2-Llama-3.1-8B/lmsys_lmsys-chat-1m/k=4_numlatents=32_lr=5e-3/best/norm",
    random=False,
    hookpoints=["norm"],
    load_in_8bit=False,
    hf_token=None,
    name="k=4_numlatents=32_lr=5e-3_split=50",
    filter_bos=False,
    overwrite=["cache"],
    seed=22,
    verbose=True,
)

path = Path("/scratch/general/vast/u1307785/delphi_cache/Skywork_Skywork-Reward-V2-Llama-3.1-8B/lmsys_lmsys-chat-1m/") / run_config.name
path.mkdir(parents=True, exist_ok=True)

hookpoints, hookpoint_to_sparse_encode, model, transcode = load_artifacts(run_config)
tokenizer = AutoTokenizer.from_pretrained(run_config.model)

populate_cache(
    run_config,
    model,
    hookpoint_to_sparse_encode,
    path,
    tokenizer,
    transcode,
)