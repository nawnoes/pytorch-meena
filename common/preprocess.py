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

if __name__=='__main__':
  origin_path = '../data/conversation_data/origin'
  processed_path = '../data/conversation_data/processed'

  file_name = 'wellness.txt'




