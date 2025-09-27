import streamlit as st

def display_success_message(message: str):
    st.success(message)

def display_error_message(message: str):
    st.error(message)

def display_warning_message(message: str):
    st.warning(message)

def display_info_message(message: str):
    st.info(message)