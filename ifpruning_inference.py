import time
from typing import Dict, List

import numpy as np
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM

from src.sparsity_prediction_model import SparsityPredictor
from src.ifpruning_config import IFPRUNING_CONFIG_V1

def main():
    device = torch.device("cuda")
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    source_model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",
        device_map="cpu",
    )
    state_dict = source_model.state_dict()

    print("Building FFN and other param dicts...")
    start = time.perf_counter()
    ffn_param_dict: Dict[str, torch.Tensor] = {}
    other_param_dict = {}

    original_keys = list(state_dict.keys())
    for key in original_keys:
        if "mlp" in key:
            ffn_param_dict[key] = state_dict.pop(key).to(device)
        else:
            other_param_dict[key] = state_dict.pop(key).to(device)
    end = time.perf_counter()
    print(f"✅ Param dicts built in {end - start:.2f}s")

    del source_model
    torch.cuda.empty_cache()

    llm_config = IFPRUNING_CONFIG_V1
    inference_llm = LlamaForCausalLM(
        llm_config,
    ).bfloat16().to(device)
    inference_llm.load_state_dict(other_param_dict, strict=False)

    sparsity_predictor_model_name = "Qwen/Qwen2.5-0.5B"
    sparsity_predictor = SparsityPredictor(
        hf_model_name=sparsity_predictor_model_name,
        num_layers=llm_config.num_hidden_layers,
        ffn_dim=llm_config.intermediate_size,
        padding_idx=128004,
    ).to(device)
    sp_tokenizer = AutoTokenizer.from_pretrained(sparsity_predictor_model_name)
    dataset = datasets.load_dataset("google/IFEval")["train"]
    input_texts: List[str] = dataset["prompt"]

    for i, text in enumerate(input_texts):
        print(f"\n🧵 Sample {i + 1}")
        sp_input_ids = sp_tokenizer(text).input_ids
        input_ids = tokenizer(text).input_ids

        sp_input_ids = torch.LongTensor(sp_input_ids).view(1, -1).to(device)
        input_ids = torch.LongTensor(input_ids).view(1, -1).to(device)

        # Step 1: Run sparsity predictor
        start = time.perf_counter()
        # batch_size, num_layers, llm_config.intermediate_size
        # True/False mask for the FFN parameters. True means selection, False means pruning that dimension.
        # add [0] since the batch size is 1
        selection_mask: torch.Tensor = sparsity_predictor(sp_input_ids)[0]
        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"🔍 Sparsity prediction time: {end - start:.4f}s")

        # Step 2: Load pruned FFN weights
        start = time.perf_counter()
        for idx in range(llm_config.num_hidden_layers):
            inference_llm.model.layers[idx].mlp.gate_proj.weight.data.copy_(
                ffn_param_dict[f"model.layers.{idx}.mlp.gate_proj.weight"][selection_mask[idx]]
            )
            inference_llm.model.layers[idx].mlp.up_proj.weight.data.copy_(
                ffn_param_dict[f"model.layers.{idx}.mlp.up_proj.weight"][selection_mask[idx]]
            )
            inference_llm.model.layers[idx].mlp.down_proj.weight.data.copy_(
                ffn_param_dict[f"model.layers.{idx}.mlp.down_proj.weight"][:, selection_mask[idx]]
            )
        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"📦 FFN weight loading time: {end - start:.4f}s")

        # Step 3: Run inference
        start = time.perf_counter()
        outputs = inference_llm(input_ids=input_ids)
        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"🚀 LLM inference time: {end - start:.4f}s")


if __name__ == "__main__":
    main()


