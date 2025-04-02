import streamlit as st
import time


MESSAGES = "messages"
ROLE = "role"


st.set_page_config(page_title="DocumentGPT", page_icon="📑")
st.title("Document GPT")


if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = []


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state[MESSAGES].append({MESSAGES: message, ROLE: role})


for message in st.session_state[MESSAGES]:
    send_message(
        message=message[MESSAGES],
        role=message[ROLE],
        save=False,
    )


message = st.chat_input("Send a message to the Assistance")


if message:
    send_message(message, "human")
    time.sleep(1)
    send_message(f"You said: {message}", "ai")

    with st.sidebar:
        st.write(st.session_state)
