import warnings

from common.arg import ModelConfig
from model.meena import Meena

warnings.filterwarnings("ignore")

from transformers import BertTokenizer
import torch
from common.generate import top_k

def sentence_mask_to_max_length(token_indices, max_length, pad_token_id = 0):
    token_len = len(token_indices)
    remainder = token_len % max_length
    diff_len = max_length - remainder
    result = token_indices + [pad_token_id]*diff_len
    return result

if __name__ =="__main__":
    base_path = '..'
    # base_path = '/Users/a60058238/Desktop/dev/workspace/transformer-electra'

    log_dir = f'{base_path}/logs'
    config_path = f'{base_path}/config/meena-config.json'

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
      dropout=config.dropout_prob
    )

    checkpoint = torch.load(f'{config.checkpoint_path}/{config.model_name}.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()
    # sent = '안병훈 공정위 국제카르텔과장은 “외국계은행의 담합으로'
    sent = 'A: 최근 정싱회담은 분위기가 어땠나? [SEP] B: 나는 정치를 잘 몰라서 [SEP] A:'
    padd_token_id = tokenizer.pad_token_id
    tokenized_sentence = tokenizer.encode(sent,add_special_tokens=False)
    while 1:
      input_ids = sentence_mask_to_max_length([tokenizer.cls_token_id,]  + tokenized_sentence,128,0)
      input_ids = torch.tensor(input_ids).unsqueeze(0)
      inputs_mask = input_ids != 0

      output = model(input_ids, input_ids, inputs_mask.unsqueeze(1), input_ids)
      pred = output[0]
      next_token_pred = pred.squeeze()[len(tokenized_sentence)]
      top_k_sample = top_k(next_token_pred,8)
      # gen = tokenizer.decode(top_k_sample).replace(' ','')
      tokenized_sentence = tokenized_sentence+top_k_sample.tolist()
      # if gen == '[SEP]':
      #     pass
      #
      # if '##'in gen:
      #   sent += gen.replace('##','')
      # else:
      #   sent += ' '+gen
      print(tokenizer.decode(tokenized_sentence,skip_special_tokens=True))
      # tokenized_sentence = tokenizer.encode(sent, add_special_tokens=False)

