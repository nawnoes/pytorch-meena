import os
import torch
import logging
import torch.nn.functional as F
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
    self.docs = []
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
        line_ids = self.tokenizer(line, add_special_tokens=False, pad_to_max_length=False)
        
        
        self.target.append(line_ids)
        
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

def make_seq2seq_data(tokenizer, dir_path, max_len):
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
    for line in tqdm(data_file,
                     desc='Data load for pretraining',
                     position=1, leave=True):
      line = line[:-1]
      line_ids = tokenizer(line, add_special_tokens=False, pad_to_max_length=False)
      
      if target_sum_len + len(line_ids)<max_len:
        target.append(line_ids)
        target_sum_len += len(line_ids)
      else:
        target_pop = target.pop(0)
        source.append(target_pop)
        
        target_sum_len -= len(target_pop)
        source_sum_len += len(target_pop)
      
      while source_sum_len>max_len:
        source_pop = source.pop(0)
        source_sum_len -= len(source_pop)
        
        
        
if __name__ == '__main__':
  data_path = '../data/plain/'
  tokenizer = BertTokenizer('../data/vocab-v1.txt')
  dataset = DatasetForSeq2seq(tokenizer, 128, data_path)

  print(dataset)
