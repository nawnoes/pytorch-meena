import os
import torch
import logging
from transformers import BertTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm


class DatasetForMeena(Dataset):
  def __init__(self, tokenizer, max_len, dir_path):
    logging.info('Start pretraining data load!')

    self.tokenizer = tokenizer
    self.max_len = max_len
    self.docs = []

    # 파일 리스트
    file_list = os.listdir(dir_path)

    # num_lines = sum(1 for line in open(path, 'r',encoding='utf-8'))
    file_progress_bar = tqdm(file_list, position=0, leave=True, bar_format='{l_bar}{bar:10}{r_bar}')
    for file_name in file_progress_bar:
      path = f'{dir_path}/{file_name}'
      data_file = open(path, 'r', encoding='utf-8')
      for line in tqdm(data_file,
                       desc='Data load for pretraining',
                       position=1, leave=True):
        line = line[:-1]
        self.docs.append(line)
    logging.info('Complete data load')

  def _tokenize_input_ids(self, input_ids: list, add_special_tokens: bool = False, pad_to_max_length: bool = True):
    inputs = torch.tensor(
      self.tokenizer.encode(input_ids, add_special_tokens=add_special_tokens, max_length=self.max_len,
                            pad_to_max_length=pad_to_max_length, return_tensors='pt', truncation=True))
    return inputs

  def __len__(self):
    return len(self.docs)

  def __getitem__(self, idx):
    inputs = self._tokenize_input_ids(self.docs[idx], pad_to_max_length=True)
    labels = inputs.clone()

    inputs = inputs.squeeze()
    labels = labels.squeeze()
    inputs_mask = inputs != 0

    return inputs, inputs_mask.unsqueeze(0), labels

class DatasetForSeq2seq(Dataset):
  def __init__(self, tokenizer, max_len, dir_path):
    logging.info('Start pretraining data for seq2seq load!')

    self.tokenizer = tokenizer
    self.max_len = max_len

    self.source = []
    self.target = []

    # 파일 리스트
    file_list = os.listdir(dir_path)

    file_progress_bar = tqdm(file_list, position=0, leave=True, bar_format='{l_bar}{bar:10}{r_bar}')
    for file_name in file_progress_bar:
      path = f'{dir_path}/{file_name}'
      data_file = open(path, 'r', encoding='utf-8')
      for line in tqdm(data_file,
                       desc='Data load for pretraining',
                       position=1, leave=True):
        line = line[:-1]
        splited_line = line.split('\t')

        self.source.append(splited_line[0])
        self.target.append(splited_line[1])

    logging.info('Complete data load')

  def _tokenize_input_ids(self, input_ids: list, add_special_tokens: bool = False, pad_to_max_length: bool = True):
    inputs = torch.tensor(
      self.tokenizer.encode(input_ids, add_special_tokens=add_special_tokens, max_length=self.max_len,
                            pad_to_max_length=pad_to_max_length, return_tensors='pt', truncation=True))
    return inputs

  def __len__(self):
    return len(self.source)

  def __getitem__(self, idx):
    encoder_input_ids = self._tokenize_input_ids(self.source[idx])
    decoder_input_ids = self._tokenize_input_ids(self.target[idx])
    labels = decoder_input_ids.clone()

    encoder_input_ids = encoder_input_ids.squeeze()
    decoder_input_ids = decoder_input_ids.squeeze()
    labels = labels.squeeze()
    encoder_inputs_mask = encoder_input_ids != 0

    return encoder_input_ids, decoder_input_ids, encoder_inputs_mask.unsqueeze(0), labels

def make_seq2seq_data(tokenizer, dir_path, max_len):
  max_len -= 1 # [CLS] 토큰을 위함
  source = []
  target = []

  source_sum_len = 0
  target_sum_len = 0
  

  # 파일 리스트
  file_list = os.listdir(dir_path)

  file_progress_bar = tqdm(file_list, position=0, leave=True, bar_format='{l_bar}{bar:10}{r_bar}')
  for file_name in file_progress_bar:
    path = f'{dir_path}/{file_name}'
    data_file = open(path, 'r', encoding='utf-8')
    out_data_file = open(f'{dir_path}processed/{file_name}', 'a', encoding='utf-8')
    for line in tqdm(data_file,
                     desc='Data load for pretraining',
                     position=1, leave=True):
      line = line[:-1]
      line_ids = tokenizer.encode(line, add_special_tokens=False, pad_to_max_length=False, max_length=max_len,truncation=True)
      
      if line == '':
        source = []
        target = []
  
        source_sum_len = 0
        target_sum_len = 0
        continue
      line += ' [SEP] '
      line_ids += [tokenizer.sep_token_id]
      if target_sum_len + len(line_ids)+1<max_len:
        target.append((line,line_ids))
        target_sum_len += len(line_ids)+1
      else:
        while target_sum_len + len(line_ids)+1 > max_len and len(target)>0:
          save_train_data(out_data_file, source, target)

          target_pop = target.pop(0)
          source.append(target_pop)

          target_sum_len -= len(target_pop[1])+1
          source_sum_len += len(target_pop[1])+1

      while source_sum_len > max_len:
        source_pop = source.pop(0)
        source_sum_len -= len(source_pop[1])+1

      # if source_sum_len >0 and target_sum_len > 0:
        # save_train_data(out_data_file, source, target)
        
def save_train_data(outfile_writer, source, target):
  if len(source) == 0 or len(target) == 0:
    return
  full_source_str = '[CLS] '
  full_target_str = '[CLS] '
  for line in source:
    full_source_str += line[0]
  for line in target:
    full_target_str += line[0]
  
  outfile_writer.write(f'{(full_source_str.strip())}\t{full_target_str.strip()}\n')
        

if __name__ == '__main__':
  data_path = '../data/plain_data/'
  tokenizer = BertTokenizer('../data/vocab-10K.txt', do_lower_case=False)
  dataset = make_seq2seq_data(tokenizer, data_path, 128)
  # dataset = DatasetForSeq2seq(tokenizer,128, data_path)
  print(dataset)
