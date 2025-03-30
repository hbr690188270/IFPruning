import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM


class SparsityPredictor(nn.Module):
    def __init__(
        self,
        hf_model_name: str,
        hidden_dim: int,
        num_layers: int,
        ffn_dim: int,
        padding_idx: int,
    ):
        """
        The sparsity predictor for IFPruning.
        Args:
            hidden_dim: the hidden dimension of the source LLM to be pruned.
            num_layers: the number of transformer layers of the source LLM.
            ffn_dim: the FFN intermediate dimension of the source LLM.
            padding_idx: the index of the padding token.
        """
        super().__init__()
        self.source_llm_num_layers = num_layers
        self.source_llm_ffn_dim = ffn_dim

        self.feature_extractor = Qwen2ForCausalLM.from_pretrained(hf_model_name)
        self.prediction_head = nn.Linear(hidden_dim, num_layers * ffn_dim)
        self.padding_idx = padding_idx

    def topk_operator(self, importance_scores: torch.Tensor, topk: float = 1536):
        """
        Args:
            importance_scores: a tensor with shape
                [batch_size, source_llm_num_layers, source_llm_ffn_dim]
        Returns:
            selected_indices: a tnesor with shape
                [batch_size, source_llm_num_layers, top_k_indices]]
        """
        bsz, source_llm_num_layers, _ = importance_scores.size()
        importance_scores = importance_scores.view(-1, importance_scores.size(-1))
        topk_indices = torch.topk(importance_scores, dim=1, k=topk, sorted=False).indices
        topk_indices = topk_indices.view(bsz, source_llm_num_layers, topk)
        return topk_indices

    def forward(self, input_ids):
        outputs = self.feature_extractor(input_ids=input_ids)
        hidden_states = outputs.hidden_states

        non_pad_mask = input_ids != self.padding_idx
        last_non_pad_indices = non_pad_mask.sum(dim=1) - 1

        batch_size = input_ids.size(0)
        last_hidden_states = hidden_states[torch.arange(batch_size), last_non_pad_indices, :]

        predictions = self.prediction_head(last_hidden_states)
        # batch_size, source_llm_num_layers, source_llm_ffn_dim
        ffn_importance_scores = predictions.view(batch_size, -1, self.source_llm_ffn_dim)

        ffn_selected_indices = self.topk_operator(ffn_importance_scores)
        return ffn_selected_indices


