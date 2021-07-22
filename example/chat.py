import torch
from common.arg import ModelConfig
from model.meena_v3 import Meena
from transformers import BertTokenizer
from common.generate import top_p, top_k


def get_encoder_input(tokenizer:BertTokenizer, input_str:list, config: ModelConfig):
    encoder_input = torch.tensor(tokenizer.encode(input_str,
                                                    add_special_tokens=False,
                                                    max_length=config.max_seq_len,
                                                    return_tensors='pt',
                                                    truncation=True))
    encoder_input_mask = encoder_input !=0
    return encoder_input, encoder_input_mask

def get_decoder_input(tokenizer:BertTokenizer, input_str:list, config: ModelConfig):
    decoder_input = torch.tensor(tokenizer.encode(input_str,
                                                    add_special_tokens=False,
                                                    max_length=config.max_seq_len,
                                                    return_tensors='pt',
                                                    truncation=True))
    return decoder_input

def get_next_token(tokenizer, logit, output_str, func):
    output_ids = tokenizer.encode(output_str,add_special_tokens=False)
    next_token_ebedd = logit.squeeze()[len(output_ids)]
    sampled_word = func(next_token_ebedd, 8)

    return sampled_word

def main():

    config_path = '../config/meena-config.json'
    checkpoint_path = '../checkpoint/komeena-3epoch-42.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = ModelConfig(config_path).get_config()

    tokenizer = BertTokenizer(config.vocab_path, do_lower_case= False)

    model = Meena(vocab_size=tokenizer.vocab_size,
                  dim= config.dim,
                  encoder_depth= config.encoder_depth,
                  decoder_depth= config.decoder_depth,
                  max_seq_len= config.max_seq_len,
                  head_num=config.n_head,
                  dropout=config.dropout_prob)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint

    model.eval()

    # Meta data for conversation
    meta_data ='[CLS] 여가와 오락 (유흥, 취미, 관심사, 휴일 활동, 동아리, 동호회) [SEP] '
    meta_data +='A=20대 여성 [SEP] '
    meta_data +='B=20대 여성 [SEP] '

    # Start of chat with Meena
    target_str = 'B: '
    print('😁 고미나에게 말을 건네세요!')

    while True:
        open_conv_query = input('A: ')
        open_conv_query = f'A: {open_conv_query}'
        source_data = meta_data + open_conv_query + ' [SEP] '
        source_input_ids, source_input_mask = get_encoder_input(tokenizer=tokenizer, input_str=source_data, config=config)
        target_input_ids = get_decoder_input(tokenizer=tokenizer, input_str=target_str, config=config)

        for _ in range(config.max_seq_len):
            logit, _ = model(source_input_ids, target_input_ids, source_input_mask)
            sampled_word = get_next_token(tokenizer, logit, target_str, top_p)
            if sampled_word == '[SEP]':
                pass
            else:


if __name__=='__main__':
    main()







