from typing import List
import numpy as np
import torch
import datasets
from transformers import AutoTokenizer, LlamaForCausalLM

from src.sparsity_prediction_model import SparsityPredictor
from src.ifpruning_config import IFPRUNING_CONFIG_V1

def main():
    device = torch.device("cuda")
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    source_model = LlamaForCausalLM.from_pretrained(model_name)
    state_dict = source_model.state_dict()

    ffn_param_dict = {}
    other_param_dict = {}

    original_keys = state_dict.keys()
    for key in original_keys:
        if "mlp" in key:
            ffn_param_dict[key] = state_dict.pop(key)
        else:
            other_param_dict[key] = state_dict.pop(key)

    llm_config = IFPRUNING_CONFIG_V1
    inference_llm = LlamaForCausalLM(llm_config)
    inference_llm.load_state_dict(other_param_dict, strict=False)

    sparsity_predictor_model_name = "Qwen/Qwen2.5-0.5B"
    sparsity_predictor = SparsityPredictor(
        hf_model_name=sparsity_predictor_model_name,
        hidden_dim=llm_config.hidden_size,
        num_layers=llm_config.num_hidden_layers,
        ffn_dim=llm_config.intermediate_size,
        padding_idx=128004,
    )
    sp_tokenizer = AutoTokenizer.from_pretrained(sparsity_predictor_model_name)
    dataset = datasets.load_dataset("google/IFEval")["train"]
    input_texts: List[str] = dataset["prompt"]

    for text in input_texts:
        sp_input_ids = sp_tokenizer(text).input_ids
        input_ids = tokenizer(text).input_ids

        sp_input_ids = torch.LongTensor(sp_input_ids).view(1, -1)
        input_ids = torch.LongTensor(input_ids).view(1, -1)

        selected_indices = sparsity_predictor(sp_input_ids)
        






