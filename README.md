# KoMeena
Korean Meena for open domain conversation

## Model
Use my own vanilla transformer model.

### Meena Train Detail
Final Meena dataset contain 341GB of Text

## Vocab
Meena's vocab sentencepiece library. Meena team use a vocabulary of **8K BPE**, which they found in early experiments to be sufficient for generating specific response. 
This project use **10K or 22K BPE**. 

## Data
34GB Korean corpus data 

## Issue
- seq2seq 모델에서 Loss가 0으로 수렴하는 문제.
    1. 인코딩 문제인지?
    2. encdoer input과 decoder input이 동일해서 발생하는것으로 예상
        - seq2seq 용으로 context를 encoder에 그 다음 문장들을 decoder 입력으로 넣어주기 위한 dataset(DatasetForSeq2seq) 테스트
        - `train/run_pretraining_v3` 로 테스트
  
## Test
작은 모델로 Sample Data에 대해 테스트

- Only Decoder Model
  using only transformer
  + 테스트 체감상 가장 좋음.
    
- Seq2Seq Model 
  seq2seq model like original transformer translation
  
- Stack Model
  Stack Model 1 Encoder + 12 Decoder 
  + 체감상 나쁘지 않음.