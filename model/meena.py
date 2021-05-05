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
    # self.encoders = clones(Encoder(d_model=dim, head_num=head_num, dropout=dropout), encoder_depth)
    # self.decoders = clones(Decoder(d_model=dim, head_num=head_num, dropout=dropout), decoder_depth)
    self.encoders = nn.ModuleList([Encoder(d_model=dim, head_num=head_num, dropout=dropout) for _ in range(encoder_depth)])
    self.decoders = nn.ModuleList([Decoder(d_model=dim, head_num=head_num, dropout=dropout) for _ in range(decoder_depth)])

    self.norm = nn.LayerNorm(dim)
    self.lm_head = nn.Linear(dim, vocab_size, bias=False)

  def forward(self, input_ids, input_mask, labels=None):
    x = self.token_emb(input_ids)
    x = x + self.position_emb(input_ids).type_as(x) # encoder input
    target = x.clone() # decoder input

    for encoder in self.encoders:
      x = encoder(x, input_mask)

    # encoder_logit = x.clone()

    for decoder in self.decoders:
      # target, encoder_output, encoder_mask)
      target = decoder(target, x, input_mask)

    lm_logits = self.lm_head(target)

    loss = None
    if labels is not None:
      # Shift so that tokens < n predict n
      shift_logits = lm_logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()

      # Flatten the tokens
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return lm_logits, loss
