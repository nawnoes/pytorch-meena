import random
import torch
import torch.nn.functional as F

import random
import torch
import torch.nn.functional as F


def random_sampling(predict, vocab, k=1024):
  # k개 중 랜덤으로 선택된 값을 반환. 사실상 top_k와 같은 기
  gen = []

  probs, indexs = torch.topk(predict, k=k, dim=-1)
  probs = probs.squeeze().tolist()[-1]
  indexs = indexs.squeeze().tolist()[-1]

  for i in range(len(indexs)):
    gen.append((vocab.to_tokens(indexs[i]), probs[i]))
  # print('topk word and value: ', gen)

  rand_num = random.randint(0, k - 1)
  gen_word = vocab.to_tokens(indexs[rand_num])

  return gen_word


def top_p(logits, threshold=0.9, is_uniform_sample=False):
  sorted_logits, sorted_indices = torch.sort(logits, descending=True)
  indexs = sorted_indices.tolist()

  sorted_softmax_logits = torch.softmax(sorted_logits, dim=-1)
  cum_probs = torch.cumsum(sorted_softmax_logits, dim=-1)

  sorted_indices_to_remove = cum_probs > threshold
  top_p_index = 0

  # Top-p에 해당하는 index를 획득
  for i in range(len(sorted_indices_to_remove)):
    if sorted_indices_to_remove[i] == True:
      top_p_index = 0 if i == 0 else i - 1
      break

  if is_uniform_sample:
    # uniform sampling
    sampled_index = random.randint(0, top_p_index - 1)
  else:
    # sampling by probability
    # 확률에 따라 샘플링 된 인덱스를 반환
    sampled_index = torch.multinomial(sorted_softmax_logits, 1)

  return indexs[sampled_index]


def top_k(predict, k, is_uniform_sample=False):
  # topk 중 랜덤으로 선택된 값을 반환.
  probs, indexs = torch.topk(predict, k=k, dim=-1)

  if is_uniform_sample:
    # uniform sampling
    sampled_index = random.randint(0, k - 1)
  else:
    # sampling by probability
    # 확률에 따라 샘플링 된 인덱스를 반환
    sampled_index = torch.multinomial(probs, 1)

  return indexs[sampled_index]

def sample_and_rank(logit, N, temperature=0.88, is_uniform_sample=True):
  logit = logit.squeeze()
  logit = logit/temperature
  softmax_logit = torch.softmax(logit,dim=-1)
  # 1. Sample N independent candidate responses using plain random sampling with Temperature
  if is_uniform_sample:
    sampled_indice = torch.multinomial(logit, N)
    sampled_values = logit[sampled_indice]
  else:

  # 2. Select candidate response with highest probability
  candidate_list = list(zip(sampled_indice, sampled_values))
  max_candidate = max(candidate_list, key=lambda x: x[1])

  return max_candidate[0] # return index