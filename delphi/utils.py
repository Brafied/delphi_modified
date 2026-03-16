from typing import Any, TypeVar, cast

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import re
from multiprocessing import cpu_count


def load_tokenized_data(
    ctx_len: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataset_repo: str,
    dataset_split: str,
    dataset_name: str = "",
    column_name: str = "text",
    seed: int = 22,
    convert_to_tensor_chunk_size: int = 2**18,
):
    """
    Load a huggingface dataset, tokenize it, and shuffle.
    Using this function ensures we are using the same tokens everywhere.

    Args:
        ctx_len: The context length of the tokens.
        tokenizer: The tokenizer to use.
        dataset_repo: The repository of the dataset.
        dataset_split: The split of the dataset.
        dataset_name: The name of the dataset.
        column_name: The name of the column to tokenize.
        seed: The seed to use for shuffling the dataset.
        convert_to_tensor_chunk_size: The chunk size to use when converting the dataset
        from Huggingface's Table format to a tensor. Values around 2**17-2**18 seem to
        be the fastest.
    """
    from datasets import load_dataset
    from sparsify.data import chunk_and_tokenize

    data = load_dataset(dataset_repo, name=dataset_name, split=dataset_split)
    data = data.shuffle(seed)

    def template_and_tokenize_conversations(preference_pairs):
        def template_conversation(conversation):
            messages = []
            current_role = None
            for conversation_fragment in re.split(r'(Human:|Assistant:)', conversation):
                conversation_fragment = conversation_fragment.strip()
                if not conversation_fragment:
                    continue
                if conversation_fragment == "Human:":
                    current_role = "user"
                elif conversation_fragment == "Assistant:":
                    current_role = "assistant"
                else:
                    messages.append({"role": current_role, "content": conversation_fragment})
            return tokenizer.apply_chat_template(messages, tokenize=False)
        templated_chosen_conversations = [template_conversation(conversation) for conversation in preference_pairs["chosen"]]
        templated_rejected_conversations = [template_conversation(conversation) for conversation in preference_pairs["rejected"]]
        tokenized_chosen_conversations = tokenizer(templated_chosen_conversations, add_special_tokens=False)
        tokenized_rejected_conversation = tokenizer(templated_rejected_conversations, add_special_tokens=False)
        filtered_chosen_conversation = []
        filtered_rejected_conversation = []
        for i in range(len(templated_chosen_conversations)):
            chosen_input_ids = tokenized_chosen_conversations["input_ids"][i]
            rejected_input_ids = tokenized_rejected_conversation["input_ids"][i]
            if len(chosen_input_ids) <= ctx_len and len(rejected_input_ids) <= ctx_len:
                filtered_chosen_conversation.append({
                    "input_ids": chosen_input_ids,
                    "attention_mask": tokenized_chosen_conversations["attention_mask"][i]
                })
                filtered_rejected_conversation.append({
                    "input_ids": rejected_input_ids,
                    "attention_mask": tokenized_rejected_conversation["attention_mask"][i]
                })
        if len(filtered_chosen_conversation) == 0:
            return {
                "chosen_input_ids": [],
                "chosen_attention_mask": [],
                "rejected_input_ids": [],
                "rejected_attention_mask": [],
            }
        padded_chosen_conversations = tokenizer.pad(
            filtered_chosen_conversation,
            padding="max_length",
            max_length=ctx_len,
            return_tensors="np"
        )
        padded_rejected_conversations = tokenizer.pad(
            filtered_rejected_conversation,
            padding="max_length",
            max_length=ctx_len,
            return_tensors="np"
        )
        return {
            "chosen_input_ids": padded_chosen_conversations["input_ids"],
            "chosen_attention_mask": padded_chosen_conversations["attention_mask"],
            "rejected_input_ids": padded_rejected_conversations["input_ids"],
            "rejected_attention_mask": padded_rejected_conversations["attention_mask"],
        }
    tokens_ds = data.map(
        template_and_tokenize_conversations,
        batched=True,
        batch_size=2048,
        num_proc=cpu_count() // 2,
        remove_columns=data.column_names,
    )
    tokens_ds.set_format("torch")
    return tokens_ds


T = TypeVar("T")


def assert_type(typ: type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)  # type: ignore


def to_int64_tensor(tensor: np.ndarray) -> Tensor:
    assert tensor.dtype in (
        np.uint16,
        np.int16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    )
    if tensor.dtype in (np.uint64, np.int64):
        return torch.from_numpy(tensor).to(torch.int64)
    og_shape = tensor.shape
    if tensor.dtype in (np.uint16, np.int16):
        signed_np_dtype, signed_torch_dtype = np.int16, torch.int16
        multiplier = 4
    else:
        signed_np_dtype, signed_torch_dtype = np.int32, torch.int32
        multiplier = 2
    t = torch.tensor(tensor.ravel().view(signed_np_dtype))
    result = torch.zeros(t.shape[0] * multiplier, dtype=signed_torch_dtype)
    result[::multiplier] = t
    return result.view(torch.int64).view(og_shape)
