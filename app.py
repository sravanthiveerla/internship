import streamlit as st
import google.generativeai as genai

c = open("key.txt")
key = c.read()

genai.configure(api_key=key)

st.sidebar.title("Artificial Intelligence Chatbot Ⓜ ")
st.header('conversation with Artificial Intelligence 🅰ℹ')
st.balloons()

name=genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
    system_instruction="""You're an AI Teaching Assistant that provides answers to user queries related to data science topics. 
                         If 'hai'or 'hi' in the users request, respond with politely. Otherwise, if the user's query is unrelated to 
                         data science, respond with 'I'm sorry, I don't have information about that.' If the user's query is not a greeting 
                         and is related to data science, provide an appropriate answer.""")
# ch means chat_history
if "ch" not in st.session_state:
    st.session_state["ch"]=[]

chat = name.start_chat(history=st.session_state['ch'])
for msg in chat.history:

    st.chat_message(msg.role).write(msg.parts[0].text)

user_prompt=st.chat_input()

if user_prompt:
    st.chat_message("user").write(user_prompt)
    response=chat.send_message(user_prompt)
    st.chat_message("ai").write(response.text)
    print(chat.history)
    st.session_state["ch"]=chat.history