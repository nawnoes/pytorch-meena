# Meena 
Implementation of Meena for open domain conversation using pytorch.  
The model in this repository use vanilla Transformer seq2seq model (not Evolved Transformer). The model consists of 1 encoder and 9 decoder.  

## Model
Transformer seq2seq model.
![](./images/meena_architecture.png)

### Model Summary
- Total Prameters: **1.1B**(1,100,293,120)
```
-----------------------------------------------------------------------
      Layer (type)        Output Shape         Param #     Tr. Param #
=======================================================================
    MeenaEncoder-1      [1, 128, 2560]     104,604,160     104,604,160
    MeenaDecoder-2      [1, 128, 2560]     970,083,840     970,083,840
       LayerNorm-3      [1, 128, 2560]           5,120           5,120
          Linear-4     [1, 128, 10000]      25,600,000      25,600,000
=======================================================================
Total params: 1,100,293,120
Trainable params: 1,100,293,120
Non-trainable params: 0
-----------------------------------------------------------------------
```


## Vocab
This repository use **10K Wordpiece BPE**.
> Original Meena use sentencepiece library. Meena team use a vocabulary of 8K BPE because they found in early experiments to be sufficient for generating specific response. 


## Data
**Pretraining Data**  
- 34GB Korean corpus data 

**Test Pretrainig Data**  
- Kowiki(634MB) + Messanger data(63MB)

**Conversation Data**
- Korean SNS data (500MB)

## Pretraining
Pretrained on 34GB Korean corpus data. 
### Train Loss
- epoch: 1
- step: 2250000

![](./images/meena_pretrain_losses.png)

### Evaluation
- Total eval loss: 2.2944
- Total eval perplexity: 10.4958

```sh
2021-09-02 16:49:49.942686 | Step: 1557220 | Eval Loss: 2.294469305341254 | Perplexity: 10.495867182863075
```

## Fine-tuning
Fine-tuned on 500MB Korean SNS data

**Evaluation**

|  epoch  |  loss  | Perplexity |
|:-------:|--------|------------|
|    1    |  2.40  |    11.65   |
|    2    |  2.36  |    11.16   |
|    3    |  2.34  |    10.92   |
|    4    |  2.32  |    10.77   |



## Checkpoint
### 1. Pretrained Meena
- it's preparing
### 2. Fine-tuned Meena
- it's preparing

## Device
- V100, 16G Memory
- Cuda 10.1, Driver 418.67

## Chat Example
- Top-p sampling (threshold=0.9, min_len=15, temperature = 0.9)
### example 1
```
#### Example
Meena에게 말을 건네세요: 
User : 오늘은 날이 좋아서 밖에 나가고 싶다
Meena:
 ㅋㅋㅋㅋㅋㅋㅋㅋ
 기운내라 ㅋㅋㅋ
User : 너는 뭐하고 있니
Meena:
 유튜브보고있어
 지금
 티비보면서
User : 티비도 보고 유튜브도 보다니 대단하네
Meena:
 어 구독자
 천만은 못 넘네
 삼성화재 - 케티 간 거라 헷갈린다..
User : 지금 몇시야?
Meena:
 지금
 < 사진 >
 이거 하고이쓰게
 < 이름 > 안뇽
```
### example 2
```
Meena에게 말을 건네세요: 
User : 나 최근에 제주도로 여행 갔다 왔는데 좋더라
Meena:
 아 난 좋아
 일단은 시골 의사이신 분들 만나는 거 재밌었음
 나도 그것도 꽂혀서
 간거같음
User : 시골 의사 분들이랑 만나면 뭘하는데?
Meena:
 집에서 쉬고 싶어
 이제 막 시작해서...
User : 그래그래.. 요즘 여행 갈만한데 없나
Meena:
 항공권이 지금 40만원대야요 ㅈㅁㅌ
User : 어우 돈 꽤나 많이 들겠네
Meena:
 근데 사실 나는 강도가 약해서
 토할거같아
```

### example 3
```
Meena에게 말을 건네세요: 
User : 요즘 친구한테 안좋은 일이 있대
Meena:
 어떤?
 입대에 대한?
 어제도 말하고?
User : 군대 말고 여자친구랑 안좋은가봐
Meena:
 군대ㅋㅋㅋ아냐
 좋기는한거갘은데
 흠
 동시에 깊이 생각못할듯
User : 어떻게 위로해주는게 좋으려나?
Meena:
 우선 시작을 해봐야할것 같아 부족할것 같아서
```
