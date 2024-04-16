import streamlit as st
from openai import OpenAI

st.title('GenerateAI App - ☺ AI Code Review')
st.header('Code Review ✒')

f = open('open_key.txt')
OPENAI_API_KEY = f.read()
client = OpenAI(api_key = OPENAI_API_KEY)

query = st.text_area('📝 Enter The Code Here 📝:')
if st.button('CLICK HERE 🖱'):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "analyze the submitted code and identify potential bugs, errors, or areas of improvement"},
            {"role": "user", "content": query}
        ]
    )
    st.write(response.choices[0].message.content)