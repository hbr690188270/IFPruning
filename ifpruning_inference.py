import csv
import os
import time
from typing import Dict, List

import numpy as np
import torch
import datasets
from absl import app, flags
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    LlamaForCausalLM,
)

from src.sparsity_prediction_model import SparsityPredictor
from src.ifpruning_config import IFPRUNING_CONFIG_V1

flags.DEFINE_integer(
    "input_length",
    None,
    help="The input length for LLM generation. All inputs will be padded to this value.",
    required=True,
)
flags.DEFINE_integer(
    "output_length",
    None,
    help=(
        "The output length for LLM generation."
        " The LLM is required to generation until this length.",
    ),
    required=True,
)

FLAGS = flags.FLAGS

def main():
    csv_file = "logs/IFPruning.csv"

    device = torch.device("cuda")
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.pad_token_id = 128004

    source_model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",
        device_map="cpu",
    )
    source_model_config = AutoConfig.from_pretrained(
        model_name,
    )
    state_dict = source_model.state_dict()

    ffn_param_dict: Dict[str, torch.Tensor] = {}
    other_param_dict = {}

    original_keys = list(state_dict.keys())
    for key in original_keys:
        if "mlp" in key:
            ffn_param_dict[key] = state_dict.pop(key).to(device)
        else:
            other_param_dict[key] = state_dict.pop(key).to(device)

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
        num_layers=source_model_config.num_hidden_layers,
        ffn_dim=source_model_config.intermediate_size,
        padding_idx=128004,
    ).bfloat16().to(device)
    total_params = sum(p.numel() for p in sparsity_predictor.parameters())
    print(f"Total parameters of sparsity_predictor: {total_params:,}")

    sp_tokenizer = AutoTokenizer.from_pretrained(sparsity_predictor_model_name)
    dataset = datasets.load_dataset("google/IFEval")["train"]
    input_texts: List[str] = dataset["prompt"]

    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=FLAGS.output_length,
        min_new_tokens=FLAGS.output_length,
        eos_token_id=None,
    )


    ttft_list = []
    generation_time_list = []
    sparsity_encoding_time_list = []
    FFN_param_load_time_list = []

    for i, text in enumerate(input_texts):
        print(f"\n🧵 Sample {i + 1}")
        sp_input_ids = sp_tokenizer(text, padding="max_length", max_length=2000, truncation=True).input_ids
        input_ids = tokenizer(text, padding="max_length", max_length=2000, truncation=True).input_ids

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
        sparsity_encoding_time_list.append(end - start)

        # Step 2: Load pruned FFN weights
        torch.cuda.synchronize()
        start = time.perf_counter()
        for idx in range(llm_config.num_hidden_layers):
            inference_llm.model.layers[idx].mlp.gate_proj.weight.data.copy_(
                ffn_param_dict[f"model.layers.{idx}.mlp.gate_proj.weight"][selection_mask[idx]][:llm_config.intermediate_size, :]
            )
            inference_llm.model.layers[idx].mlp.up_proj.weight.data.copy_(
                ffn_param_dict[f"model.layers.{idx}.mlp.up_proj.weight"][selection_mask[idx]][:llm_config.intermediate_size, :]
            )
            inference_llm.model.layers[idx].mlp.down_proj.weight.data.copy_(
                ffn_param_dict[f"model.layers.{idx}.mlp.down_proj.weight"][:, selection_mask[idx]][:, :llm_config.intermediate_size]
            )
        torch.cuda.synchronize()
        end = time.perf_counter()
        FFN_param_load_time_list.append(end - start)

        torch.cuda.synchronize()
        start_prefill = time.perf_counter()
        _ = inference_llm(input_ids=input_ids)
        torch.cuda.synchronize()
        end_prefill = time.perf_counter()
        prefill_time = end_prefill - start_prefill
        ttft_list.append(prefill_time)

        torch.cuda.synchronize()
        start_gen = time.perf_counter()
        _ = inference_llm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
        )
        torch.cuda.synchronize()
        end_gen = time.perf_counter()
        total_gen_time = end_gen - start_gen

        # Step 3: Isolate generation time
        actual_generation_time = total_gen_time - prefill_time
        generation_time_list.append(actual_generation_time)

    avg_ttft = np.mean(ttft_list)
    avg_gen_time = np.mean(generation_time_list)
    tps = FLAGS.output_length / avg_gen_time
    avg_param_loading_time = np.mean(FFN_param_load_time_list)
    avg_mask_generation_time = np.mean(sparsity_encoding_time_list)

    print("Averaget TTFT: ", avg_ttft)
    print("Averaget generation time: ", avg_gen_time)
    print("Averaget TPS: ", tps)
    print("Averaget mask generation time: ", avg_mask_generation_time)
    print("Averaget parameter loading: ", avg_param_loading_time)

    # Prepare row data
    row = {
        "input_length": FLAGS.input_length,
        "output_length": FLAGS.output_length,
        "ttft": avg_ttft,
        "generation_time": avg_gen_time,
        "tps": FLAGS.output_length / avg_gen_time,
        "mask generation": avg_mask_generation_time,
        "param loading": avg_param_loading_time,
    }

    # Check if the file exists to decide whether to write header
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"📁 Logged results to: {csv_file}")



if __name__ == "__main__":
    main()


