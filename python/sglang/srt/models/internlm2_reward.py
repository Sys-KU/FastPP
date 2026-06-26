# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed import get_pp_group

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.internlm2 import InternLM2ForCausalLM, InternLM2Model
from sglang.srt.utils import PPMissingLayer


class InternLM2ForRewardModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.model = InternLM2Model(config, quant_config)
        if get_pp_group().is_last_rank:
            self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        else:
            self.v_head = PPMissingLayer()
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
        intermediate_tensors: Optional[torch.Tensor] = None,
    ) -> EmbeddingPoolerOutput:
        assert get_embedding, "InternLM2ForRewardModel is only used for embedding"
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds, intermediate_tensors)
        if not get_pp_group().is_last_rank:
            return hidden_states
        last_token_hidden = self.pooler(hidden_states, forward_batch).embeddings
        scores = self.v_head(last_token_hidden)
        return EmbeddingPoolerOutput(scores)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return InternLM2ForCausalLM.load_weights(self, weights)


EntryClass = InternLM2ForRewardModel
