import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to our final project !! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    ### Students:
        20110127 - Nguyen Tien Dung
        20110438 - Ngo Nguyen Quoc Anh
    ### Teacher: 
        Tran Tien Duc
    ### Subject:
        Machine Learning
"""
)
