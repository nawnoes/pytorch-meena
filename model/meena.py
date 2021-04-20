from collections import namedtuple

import torch
from torch import nn
from model.util import clones, make_std_mask
import torch.nn.functional as F
from model.util import log, gumbel_sample, mask_with_tokens, prob_mask_like, get_mask_subset_with_prob
from model.transformer import PositionalEmbedding,Encoder
from transformers.activations import get_activation
from torch.nn import CrossEntropyLoss


MeenaOutput = namedtuple('MeenaOutput', [
  'lm_logits',
  'loss',
  'encoder_logit',
  'decoder_logit',
])


class Meena(nn.Module):
  def __init__(self,
               vocab_size,
               dim=512,
               encoder_depth=1,
               decoder_depth=12,
               max_seq_len=512,
               head_num=8,
               dropout=0.1):
    super(Meena, self).__init__()

    # Embedding
    self.token_emb = nn.Embedding(vocab_size, dim)
    self.position_emb = PositionalEmbedding(dim, max_seq_len)
    # Meena Model
    self.encoders = clones(Encoder(d_model=dim, head_num=head_num, dropout=dropout), encoder_depth)
    self.decoders = clones(Decoder(d_model=d_model, head_num=head_num, dropout=dropout), decoder_depth)

    self.norm = nn.LayerNorm(dim)
    self.lm_head = nn.Sequential(
      nn.Linear(dim, dim),
      nn.Linear(dim, vocab_size)
    )

  def forward(self, input_ids, input_mask):
    x = self.token_emb(input_ids)
    x = x + self.position_emb(input_ids).type_as(x)

    for encoder in self.encoders:
      x = encoder(x, input_mask)
    x = self.norm(x)

    encoder_logit = x.clone() # encoder_logit
    target_mask = make_std_mask(input_ids) # target mask 생성

    for decoder in self.decoders:
      x = decoder(x, target_mask)

    lm_logits = self.lm_head(x)

    loss = None
    if labels is not None:
      # Shift so that tokens < n predict n
      shift_logits = lm_logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()
      # Flatten the tokens
      loss_fct = CrossEntropyLoss(ignore_index=0)
      loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return MeenaOutput(lm_logits, loss, encoder_logit, x)

