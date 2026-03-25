import argparse
from pathlib import Path
from transformers import AutoTokenizer
from delphi.__main__ import load_artifacts, populate_cache
from delphi.config import ConstructorConfig, RunConfig, CacheConfig, SamplerConfig

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--sae_directory", required=True, type=str)
    argument_parser.add_argument("--output_directory", required=True, type=str)
    arguments = argument_parser.parse_args()


    cache_config = CacheConfig(
        dataset_repo="lmsys/lmsys-chat-1m",
        dataset_split="train[:50%]",
        dataset_name="",
        batch_size=325,
        cache_ctx_len=2000,
        n_splits=1,
    )
    run_config = RunConfig(
        cache_cfg=cache_config,
        constructor_cfg=ConstructorConfig(),
        sampler_cfg=SamplerConfig(),
        model="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        sparse_model=arguments.sae_directory,
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

    path = Path(arguments.output_directory) / run_config.name
    path.mkdir(parents=True, exist_ok=True)

    _, hookpoint_to_sparse_encode, model, transcode = load_artifacts(run_config)
    tokenizer = AutoTokenizer.from_pretrained(run_config.model)

    populate_cache(
        run_config,
        model,
        hookpoint_to_sparse_encode,
        path,
        tokenizer,
        transcode,
    )

if __name__ == "__main__":
    main()