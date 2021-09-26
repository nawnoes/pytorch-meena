import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.ticker as ticker


# 라인그래프
def print_json_line_graph(json_path):
  f = open(json_path, 'r')
  json_data = json.load(f)
  lists = json_data.items()  # sorted by key, return a list of tuples
  len_total_step = len(lists)
  num = 10

  lists = list(filter(lambda x: int(x[0]) % num==0, lists))
  x, y = zip(*lists)  # unpack a list of pairs into two tuples

  plt.plot(x, y, 'r')
  plt.xticks(np.arange(0,len_total_step//num+1), labels=map(lambda x: x if x % ((len_total_step//num)*num) ==0 else '',range(0,len_total_step+1,num)))
  plt.xlabel('step')
  plt.ylabel('loss')
  plt.title('Train Losses')
  plt.show()

if __name__=='__main__':
  # print_json_line_graph('../logs/komeena-base_train_results.json')
  print_json_line_graph('../logs/komeena-base-finetuning_train_results.json')