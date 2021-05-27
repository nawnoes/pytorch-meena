# coding=utf-8
# Copyright 2020 The Fairseq Authors and The HuggingFace Inc. team.
#
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
"""
Meena configuration
HuggingFace의 Bart 모델 코드를 바탕으로 일부 재작성한 Meena configuration 코드입니다.
원코드: https://github.com/huggingface/transformers/blob/master/src/transformers/configuration_bart.py
"""
import json
from typing import Type, TypeVar

from transformers.configuration_utils import PretrainedConfig


T = TypeVar("T")


class MeenaConfig(PretrainedConfig):
    r"""
    Configuration class for Meena.
    Args:
        vocab_size (:obj:`int`, optional, defaults to 32000):
            defines the different tokens that can be represented by `inputs_ids` passed to the forward method.
        d_model (:obj:`int`, optional, defaults to 768):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (:obj:`int`, optional, defaults to 2):
            Number of encoder layers
        decoder_layers (:obj:`int`, optional, defaults to 12):
            Number of decoder layers
        encoder_attention_heads (:obj:`int`, optional, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, optional, defaults to 12):
            Number of attention heads for each attention layer in the Transformer decoder.
        encoder_ffn_dim (:obj:`int`, optional, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in decoder.
        decoder_ffn_dim (:obj:`int`, optional, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in decoder.
        dropout (:obj:`float`, optional, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, optional, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, optional, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        max_position_embeddings (:obj:`int`, optional, defaults to 256):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        max_encoder_length (:obj:`int`, optional, defaults to 128):
            The maximum sequence length that the encoder might ever be used with.
        init_std (:obj:`float`, optional, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        force_bos_token_to_be_generated (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to force BOS token to be generated at step 1 (after ``decoder_start_token_id``), only true for `bart-large-cnn`.
        is_encoder_decoder (:obj:`bool`, optional, defaults to :obj:`True`):
            Whether this is an encoder/decoder model
        pad_token_id (:obj:`int`, optional, defaults to 0)
            Padding token id.
        unk_token_id (:obj:`int`, optional, defaults to 1)
            Unknown of stream token id.
        bos_token_id (:obj:`int`, optional, defaults to 2)
            Beginning of stream token id.
        eos_token_id (:obj:`int`, optional, defaults to 3)
            End of stream token id.
        sept_token_id (:obj:`int`, optional, defaults to 4)
            Turn separator of stream token id.
    """

    model_type = "meena"

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        encoder_layers: int = 2,
        decoder_layers: int = 12,
        encoder_attention_heads: int = 12,
        decoder_attention_heads: int = 12,
        encoder_ffn_dim: int = 3072,
        decoder_ffn_dim: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_position_embeddings: int = 256,
        max_encoder_length: int = 128,
        init_std: float = 0.02,
        force_bos_token_to_be_generated: bool = False,
        is_encoder_decoder: bool = True,
        pad_token_id: int = 0,
        unk_token_id: int = 1,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        sept_token_id: int = 4,
        **common_kwargs,
    ):
        r"""
        :class:`~transformers.MeenaConfig` is the configuration class for `MeenaModel`.
        """
        if "hidden_size" in common_kwargs:
            raise ValueError("hidden size is called d_model")
        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            **common_kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = self.num_hidden_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.max_encoder_length = max_encoder_length
        self.init_std = init_std

        # 3 Types of Dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout

        self.force_bos_token_to_be_generated = force_bos_token_to_be_generated

        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.sept_token_id = sept_token_id

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model

    @classmethod
    def from_json(cls: Type[T], json_file_path: str, **kwargs) -> T:
        """
        Json으로부터 Config 클래스를 생성합니다.
        """
        with open(json_file_path, "r") as f:
            return cls.from_dict(json.load(f), **kwargs)