import streamlit as st
from utils import pdf_tools
from langchain.memory import ConversationBufferMemory

st.title("📃 PDF Intelligent Analysis Tools")
with st.sidebar:
    openai_api_key = st.text_input("Please provide your api key:",type = "password")
    no_api = st.checkbox("👀 I have no api key")
    if no_api:
        st.markdown("☞ Method to get your api key: https://api.aigc369.com/register?aff=87kh")

# 初始化记忆
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(return_messages = True,
                                                          memory_key = 'chat_history',
                                                          output_key = 'answer')

uploaded_file = st.file_uploader("Please upload your file",type = "pdf")
question = st.text_input("Please ask questions about your PDF file.",disabled = not uploaded_file)# 上传的文件不是pdf，不可以使用文本框

# 如果用户没有提供API密钥

if uploaded_file and question and not openai_api_key:
    st.info("❗ Please input your openai api key")


# 用户提供了API密钥

if uploaded_file and question and openai_api_key:
    with st.spinner("🧠 The AI is accelerating its thinking......"):
        response = pdf_tools(openai_api_key , st.session_state["memory"],uploaded_file, question)
        # response 是一个字典 : answer键对应AI回答的问题   chat_history 历史消息对话列表

    st.write("💡  Answer ")
    st.write(response["answer"])
    st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("history:"):# 展示历史消息
        for i in range(0,len(st.session_state["chat_history"]),2):# 一条条的展示历史消息
            # 展示用户的问题和AI的回复
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i+1]
            st.write(human_message.content)
            st.write(ai_message.content)

            if i < len(st.session_state["chat_history"])-2:
                st.divider()

