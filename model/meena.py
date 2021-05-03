from collections import namedtuple
from torch import nn
from model.util import clones, make_std_mask
from model.transformer import PositionalEmbedding, Encoder, Decoder
from torch.nn import CrossEntropyLoss

MeenaOutput = namedtuple('MeenaOutput', [
  'lm_logits',
  'loss',
  'encoder_logit',
  'decoder_logit',
])

"""
Meena best performing model use Evolbed Transformer
  Model config
    - 1    encoder block
    - 13   decoder block
    - 2560 hidden size
    - 128  max token length 
"""


class Meena(nn.Module):
  def __init__(self,
               vocab_size,
               dim=2560,
               encoder_depth=1,
               decoder_depth=13,
               max_seq_len=128,
               head_num=32,
               dropout=0.1):
    super(Meena, self).__init__()

    # Embedding
    self.token_emb = nn.Embedding(vocab_size, dim)
    self.position_emb = PositionalEmbedding(dim, max_seq_len)

    # Meena Model
    self.encoders = clones(Encoder(d_model=dim, head_num=head_num, dropout=dropout), encoder_depth)
    self.decoders = clones(Decoder(d_model=dim, head_num=head_num, dropout=dropout), decoder_depth)

    self.norm = nn.LayerNorm(dim)
    self.lm_head = nn.Linear(dim, vocab_size, bias=False)

  def forward(self, input_ids, input_mask, labels=None):
    x = self.token_emb(input_ids)
    x = x + self.position_emb(input_ids).type_as(x)

    for encoder in self.encoders:
      x = encoder(x, input_mask)
    x = self.norm(x)

    encoder_logit = x.clone()  # encoder_logit
    # TODO 디코더에서 target_mask upper triangular matrix 수정 후 학습해볼것.
    # target_mask = make_std_mask(input_ids) # target mask 생성

    for decoder in self.decoders:
      x = decoder(x)

    lm_logits = self.lm_head(x)

    loss = None
    if labels is not None:
      # Shift so that tokens < n predict n
      shift_logits = lm_logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()

      # Flatten the tokens
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return lm_logits, loss, encoder_logit, x
