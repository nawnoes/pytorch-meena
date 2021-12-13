import torch
from model.meena import Meena
from common.arg import ModelConfig
from transformers import BertTokenizer
import pytorch_model_summary as pms

base_path = '..'
config_path = f'{base_path}/config/meena-finetuning-config-v3.json'

# Config
config = ModelConfig(config_path=config_path).get_config()

# Tokenizer
tokenizer = BertTokenizer(vocab_file=config.vocab_path, do_lower_case=False)

# Meena Model
model = Meena(
    vocab_size=tokenizer.vocab_size,
    dim=config.dim,
    encoder_depth=config.encoder_depth,
    decoder_depth=config.decoder_depth,
    max_seq_len=config.max_seq_len,
    head_num=config.n_head,
    dropout=config.dropout_prob)

# Model Inputs
encoder_input_ids = torch.randint(0,10000,(1,128))
decoder_input_ids = torch.randint(0,10000,(1,128))
encoder_input_mask = torch.tensor([True for _ in range(128)])

# Summary
pms.summary(model, encoder_input_ids,decoder_input_ids, encoder_input_mask, print_summary=True)

"""
print in console
-----------------------------------------------------------------------
      Layer (type)        Output Shape         Param #     Tr. Param #
=======================================================================
    MeenaEncoder-1      [1, 128, 2560]     104,604,160     104,604,160
    MeenaDecoder-2      [1, 128, 2560]     970,083,840     970,083,840
       LayerNorm-3      [1, 128, 2560]           5,120           5,120
          Linear-4     [1, 128, 10000]      25,600,000      25,600,000
=======================================================================
Total params: 1,100,293,120
Trainable params: 1,100,293,120
Non-trainable params: 0
-----------------------------------------------------------------------
"""