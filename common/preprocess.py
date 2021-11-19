# import pandas as pd
import os
import json
import re

def add_turn_info(origin_path, processed_path):
  file_name = 'wellness.txt'

  f = open(f'{origin_path}/{file_name}','r',encoding='utf-8')
  w_f = open(f'{processed_path}/{file_name}','w',encoding='utf-8')

  count =0
  while True:
    line = f.readline()
    if not line: break
    if line == '\n':
      count=0
      w_f.write('\n')
    else:
      if count % 2 ==0:
        w_f.write(f'A: {line}')
      else:
        w_f.write(f'B: {line}')
      count +=1

  f.close()
  w_f.close()

def csv2txt_emotional_conversation_data(origin_path, processed_path):
  file_name = 'emotional_conversation'
  df = pd.read_csv(f'{origin_path}/{file_name}.csv')

  w_f = open(f'{processed_path}/{file_name}.txt', 'w', encoding='utf-8')
  for i, row in df.iterrows():
    # for row in df.itertuples(index=False):
    for i, item in row.items():
      if not pd.isna(item):
        if i == '상황키워드':
          w_f.write(item + '\n')
        elif i not in ['번호', '연령', '성별', '신체질환', '감정_소분류', '감정_대분류']:
          if '사람문장' in i:
            w_f.write('A: ' + item + '\n')
          elif '시스템응답' in i:
            w_f.write('B: ' + item + '\n')
    w_f.write('\n')
  w_f.close()

def sns_conversation_data(file_path):
  return_data = []
  participants = {
    "P01":"A",
    "P02":"B",
    "P03":"C",
    "P04":"D",
    "P05":"E",
    "P06":"F",
    "P07":"G",
    "P08":"H",
    "P09": "I",
    "P10": "J",

  }
  with open(file_path,"r",encoding='utf-8') as json_file:
    json_data = json.load(json_file,strict=False)

  # print(json_data)
  converstions = json_data["data"]

  for conv in converstions:
    header = conv["header"]["dialogueInfo"]
    meta = f"{header['type']} {header['topic']}"
    return_data.append(meta)
    body = conv['body']
    for b in body:
      return_data.append(f'{participants[b["participantID"]]}: {b["utterance"]}')

    return_data.append("")

  return return_data

def run_preprocess_sns_data(dir_path= '/Volumes/T7 Touch/NLP Data/korean_sns'):
  origin_path = f'{dir_path}/origin'
  processed_path = f'{dir_path}/processed'

  file_list = os.listdir(origin_path)
  w_f = open(f'{processed_path}/korean_sns.txt','w',encoding='utf-8')

  for file in file_list:
    print(f'process {origin_path}/{file}')
    datas = sns_conversation_data(f'{origin_path}/{file}')
    for data in datas:
      w_f.write(data+'\n')
    # w_f.write('\n')

  w_f.close()

def find_system_token(file_path = "../data/plain/korean_sns.txt"):
  system_token = set([])
  pattern = re.compile("#@[ㄱ-ㅎ|가-힣]*#[ㄱ-ㅎ|가-힣]*#|#@[ㄱ-ㅎ|가-힣]*#")
  file = open(file_path,'r')
  while True:
    line = file.readline()
    if line is None:
      break

    results = pattern.findall(line)
    set_results = set(results)
    diff_set = set_results - system_token
    if len(diff_set) > 0:
      for item in diff_set:
        system_token.add(item)

  print(system_token)
  return system_token




if __name__=='__main__':
  data_path = '/Volumes/T7 Touch/NLP Data/korean_sns/origin/미용과건강.json'

  # data = sns_conversation_data(data_path)
  # run_preprocess_sns_data()
  # print(data)
  find_system_token()









