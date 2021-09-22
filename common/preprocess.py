import pandas as pd

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

if __name__=='__main__':
  origin_path = '../data/conversation_data/origin'
  processed_path = '../data/conversation_data/processed'

  file_name = 'emotional_conversation'
  df = pd.read_csv(f'{origin_path}/{file_name}.csv')

  w_f = open(f'{processed_path}/{file_name}.txt','w',encoding='utf-8')
  for i, row in df.iterrows():
  # for row in df.itertuples(index=False):
    for i, item in row.items():
      if not pd.isna(item):
        if i == '상황키워드':
          w_f.write(item+'\n')
        elif i not in ['번호', '연령', '성별', '신체질환', '감정_소분류','감정_대분류']:
          if '사람문장' in i:
            w_f.write('A: '+item+'\n')
          elif '시스템응답' in i:
            w_f.write('B: ' + item + '\n')
    w_f.write('\n')
  w_f.close()








