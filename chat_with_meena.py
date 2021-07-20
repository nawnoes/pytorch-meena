import torch
from common.arg import ModelConfig
from model.meena_v3 import Meena
from transformers import BertTokenizer
from common.generate import top_p, top_k

config_path ='./config/meena-config.json'
checkpoint_path = './checkpoint/komeena-3epoch-42.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = ModelConfig(config_path)

tokenizer = BertTokenizer(config.vocab_path, do_lower_case= False)

model = Meena(vocab_size=tokenizer.vocab_size,
              dim= config.dim,
              encoder_depth= config.encoder_depth,
              decoder_depth= config.decoder_depth,
              max_seq_len= config.max_seq_len,
              head_num=config.n_head,
              dropout=config.dropout_porb)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
del checkpoint

model.eval()

while True:
    input




