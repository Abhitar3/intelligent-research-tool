import streamlit as st
from rag import process_urls,generate_answer
st.title('Research Tool')

url1=st.sidebar.text_input('url 1')
url2=st.sidebar.text_input('url 2')
url3=st.sidebar.text_input('url 3')

process_url_button=st.sidebar.button('Process Urls')

placeholder=st.empty()
if process_url_button:
    urls=[url for url in (url1,url2,url3) if url!='']
    if len(urls)==0:
        placeholder.text('you must provide atlease one valid url')
    else:
        for status in process_urls(urls):
            placeholder.text(status)

query=placeholder.text_input('Enter Your Question')
if query:
    try:
        answer,sources= generate_answer(query)
        st.header("Answer")
        st.write(answer)

        if sources:
            st.subheader('Sources:')
            st.write(sources)
    except RuntimeError as e:
        placeholder.text("You must process urls")