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
    dataset = datasets.load_dataset("google/IFEval")["train"]
    input_texts: List[str] = dataset["prompt"]

    for i, text in enumerate(input_texts):
        print(f"\n🧵 Sample {i + 1}")
        input_ids = tokenizer(text).input_ids

        input_ids = torch.LongTensor(input_ids).view(1, -1).to(device)

        start = time.perf_counter()
        outputs = source_model(input_ids=input_ids)
        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"🚀 LLM inference time: {end - start:.4f}s")


if __name__ == "__main__":
    main()


