# Meena (PyTorch)

An open-domain Korean chatbot project inspired by **Meena** (seq2seq dialogue model), implemented in **PyTorch**.

Note: this repository uses a **vanilla Transformer seq2seq** architecture (not the “Evolved Transformer” used in the original Meena paper). The default configuration uses **1 encoder layer + 9 decoder layers**.

## Model
Transformer seq2seq model.

### Model Summary
- Total Parameters: **1.1B** (1,100,293,120)
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
This repository uses a **10K WordPiece (BPE-family)** vocabulary.

The original Meena paper used SentencePiece with an 8K BPE vocabulary; this project uses WordPiece as an implementation choice.


## Data
**Pretraining Data**  
- 34GB Korean corpus data 

**Test Pretraining Data**  
- Kowiki(634MB) + Messanger data(63MB)

**Conversation Data**
- Korean SNS data (500MB)

## Pretraining
Pretrained on 34GB Korean corpus data. 
### Train Loss
- epoch: 1
- step: 2250000

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

## Device
- V100, 16G Memory
- Cuda 10.1, Driver 418.67

## Configuration (snapshot)

Main config file:

- `config/meena-config.json`

Key parameters in the default config:

- `dim=2560`, `n_head=32`, `max_seq_len=128`
- `encoder_depth=1`, `decoder_depth=9`
- `vocab_path=../data/vocab-10K.txt` (10K WordPiece vocab)
- fp16 training enabled via Apex AMP (`fp16=true`, `fp16_opt_level=O2`)

## How to run

### Chat / inference

Run an interactive chat loop:

```bash
python example/chat.py
```

The script loads:

- config: `config/meena-config.json`
- a checkpoint path set inside `example/chat.py` (`checkpoint_path = ...`)

Decoding:

- top-p (nucleus) sampling is used by default (`common/generate.py`).

### Training (best-effort)

- Pretraining: `train/run_pretraining.py`
- Fine-tuning: `train/run_finetuning.py`

These scripts are written as research code and may require path edits depending on your local data layout.

## Data format notes (best-effort)

The dataset loaders under `common/dataset.py` expect a directory of text files.
Conversation files are typically structured as:

- one utterance per line
- blank lines separating conversations
- speaker prefixes like `A : ...` / `B : ...` are commonly used in this repo (see `example/chat.py`)
