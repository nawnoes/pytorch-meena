from torch import nn
from model.transformer import PositionalEmbedding, Encoder, Decoder
from torch.nn import CrossEntropyLoss


class MeenaEncoder(nn.Module):
  def __init__(self,
               token_emb,
               dim=2560,
               encoder_depth=1,
               max_seq_len=128,
               head_num=32,
               dropout=0.1):
    super().__init__()
    self.token_emb = token_emb
    self.position_emb = PositionalEmbedding(dim,max_seq_len)

    self.encoders = nn.ModuleList([Encoder(d_model=dim, head_num=head_num, dropout=dropout) for _ in range(encoder_depth)])

  def forward(self, input_ids, input_mask):
    inputs_embed = self.token_emb(input_ids)
    position_embed = self.position_emb(input_ids)

    hidden_states = inputs_embed + position_embed

    for encoder in self.encoders:
      hidden_states = encoder(hidden_states, input_mask)

    return hidden_states

class MeenaDecoder(nn.Module):
  def __init__(self,
               token_emb,
               dim=2560,
               decoder_depth=13,
               max_seq_len=128,
               head_num=32,
               dropout=0.1):
    super().__init__()
    self.token_emb = token_emb
    self.position_emb = PositionalEmbedding(dim,max_seq_len)

    self.decoders = nn.ModuleList([Decoder(d_model=dim, head_num=head_num, dropout=dropout) for _ in range(decoder_depth)])

  def forward(self, input_ids, encoder_hidden_states, encoder_mask):
    inputs_embed = self.token_emb(input_ids)
    position_embed = self.position_emb(input_ids)

    hidden_states = inputs_embed + position_embed
    for decoder in self.decoders:
      hidden_states = decoder(hidden_states, encoder_hidden_states, encoder_mask)

    return hidden_states

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

    # Meena Model
    self.meena_encoder = MeenaEncoder(self.token_emb,dim,encoder_depth,max_seq_len,head_num,dropout)
    self.meena_decoder = MeenaDecoder(self.token_emb,dim,decoder_depth,max_seq_len,head_num,dropout)

    # LM Head
    self.norm = nn.LayerNorm(dim)
    self.lm_head = nn.Linear(dim, vocab_size, bias=False)

  def forward(self, encoder_input_ids, decoder_input_ids, encoder_input_mask, labels=None):
    encoder_hidden_state = self.meena_encoder(encoder_input_ids, encoder_input_mask)
    decoder_logit = self.meena_decoder(decoder_input_ids, encoder_hidden_state, encoder_input_mask)

    lm_logits = self.lm_head(self.norm(decoder_logit))

    loss = None
    if labels is not None:
      # Shift so that tokens < n predict n
      shift_logits = lm_logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()

      # Flatten the tokens
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return lm_logits, loss
