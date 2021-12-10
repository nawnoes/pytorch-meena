import streamlit as st
import random
render_count = 0
def get_random_message()->str:
    sample_text=['만나서 반가워',
                 '안녕',
                 '오늘 날씨는 어때?',]
    return random.choice(sample_text)

def chat(input_txt:str)->str:
    pass
# def on_change_text_input(text):
#    st.session_state['chat_input'] = text
#    print(st.session_state['chat_input'])
def send_chat():
    chat_input=st.session_state['chat_input']
    print(chat_input)
    st.session_state['chat_input'] = 'Changed'
    print(st.session_state['chat_input'])

def get_text_input_container():
    txt_input = st.empty()
    txt_input.text_input('text')

st.session_state['chat_input']='please input txt'
print(f'render_count: {render_count}')
render_count += 1
# Header
st.subheader('Chat Example')
st.markdown('chat with **pytorch-meena**.')


# Chat
st.session_state['contents'] = '안녕하세요\n만나서 반가워요.\n'
chat_area = st.text_area('Chat',st.session_state['contents'],key=1)

# Text Input
placeholder = st.empty()
input = placeholder.text_input('text1')
send_bttn = st.empty()
click_clear = send_bttn.button('clear text input1')
# print(f'click_cleaner  {click_clear}')
if click_clear:
    print(input)
    st.session_state['contents'] += f"{input}\n"
    chat_area = st.text_area('Chat',st.session_state['contents'],key=2)
