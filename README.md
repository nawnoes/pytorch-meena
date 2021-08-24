# KoMeena
Korean Meena for open domain conversation

## Model
My transformer model.

## Vocab
Meena's vocab sentencepiece library. Meena team use a vocabulary of **8K BPE**, which they found in early experiments to be sufficient for generating specific response. 
This project use **10K or 22K BPE**. 

## Data
- Pretraining Data: 34GB Korean corpus data 
- Test Pretrainig Data: Kowiki(634MB) + Messanger data(63MB)

## Chat Example
### Test Pretrained Model
Model prtrained using kowiki & Messager data. 
- 3 epoch
- ./example/chat.py
#### Meta Data
  ```py
    meta_data ='[CLS] 여가와 오락 (유흥, 취미, 관심사, 휴일 활동, 동아리, 동호회) [SEP] '
    meta_data +='A=20대 여성 [SEP] '
    meta_data +='B=20대 여성 [SEP] '
  ```
#### Example
- A: User
- B: KoMeena
```txt
😁 고미나에게 말을 건네세요!
A : 날도 더운데 영화나 보러가자ㅋㅋ
B : 어디에갈까
A : cgv로 갈까??
B : 아님 지나?
A : 지나는 뭔데
B : 1986년에 발매된 롤플레잉 비디오 게임이다.
A : 으음 영화는 없니
B : 영화는 대한민국의 영화 영화, 드라마, 애니메이터 등은 배우 양성으로 듣고 있지만, 주말에 그녀는 대부분이 영화를 좋아한다.
A : 위키스러운 말을 하는구나
B : 2011년 하반기의 클럽 퀸 엘리자베스 2세 시기이다.
A : 맥락도 없구
B : 맥락이 무명이여치지
A : 배고프다 이눔아
B :
A : 왜 말 없냐
B : 으ㅓ으으으으으으으!!!!!!! ~!
A : 비명 지르지 마ㅋㅋㅋㅋㅋ
B : ㅇ0ㅇ 이전 제목이다
```

## Issue
- seq2seq 모델에서 Loss가 0으로 수렴하는 문제.
    1. 인코딩 문제인지?
    2. encdoer input과 decoder input이 동일해서 발생하는것으로 예상
        - [2021.07.05] Encoder의 입력과 Decoder의 입력을 다르게 변경해서 테스트
        - [2021.07.12] 학습 정상 동작 확인.
- 성능 문제
  **개선방법**  
    
    - ① Pretraining 후 적은 대화 데이터에 대해 Fine-tuning
    - ② 대화 데이터 수집
    
- 대화 데이터 수집 필요
    
## Test
Three type model tested 
- ① Only Decoder Model: only transformer decoder(ex. GPT)
- ② Seq2Seq Model: Seq2seq model like original transformer model
- ③ Stack Model: Stack 1 Encoder + 12 Decoder