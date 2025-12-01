import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import uuid

model_name = "cmarkea/bloomz-560m-sft-chat"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

text_pipeline = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    return_full_text=False
)

prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate creative, thought-provoking conversations, or ideas that encourage"
                       "discussion and exploration. make prompts short, engaging, for conversation"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )


def parse_pipeline(output):
    pass


def run_pipeline(prompt_value):
    full_prompt_str = prompt_value.to_string()
    output = text_pipeline(full_prompt_str)
    return parse_pipeline(output)

core_chain = prompt | text_pipeline

store = {}

def get_session_id(session_id: str):
    if session_id not in store:
        from langchain_core.chat_history import InMemoryChatMessageHistory
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chat_with_history = RunnableWithMessageHistory(
    core_chain,
    get_session_id,
    input_messages_key="question",
    history_messages_key="history",
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.display_messages = []

st.set_page_config(page_title="Python chatbot", page_icon="📎")

st.markdown(
    """
    <div style="background-color:#f1f5c4; padding: 15px; border-radius: 10px;">
        <h1 style="color:#171714; text-align: center;"> bloomz-560m Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "**cmarkea/bloomz-560m-sft-chat** open-source model."
)

st.write("---")

for msg in st.session_state.display_messages:
    role = "👤 student" if msg["role"] == "user" else "📎 Bloom_AI"
    color = "#171714"

    st.markdown(
        f"""
        <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
            <b>{role}:</b> {msg['content']}
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("---")

user_input = st.text_area(
    "Enter question here:",
    placeholder="Example: What does a Python list do?",
    key="user_input_area"
)

col1, col2 = st.columns(2)
with col1:
    send_clicked = st.button("Ask the bot")
with col2:
    clear_clicked = st.button("Clear conversation")

if clear_clicked:
    if st.session_state.session_id in store:
        del store[st.session_state.session_id]
        st.session_state.session_id = str(uuid.uuid4())

    st.session_state.display_messages = []
    st.rerun()


if send_clicked and user_input.strip():
    st.session_state.display_messages.append({"role": "user", "content": user_input})

    with st.spinner("Bigscience/560m is thinking..."):
        try:
            response = chat_with_history.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            assistant_reply = response
            st.session_state.display_messages.append(
                {"role": "assistant", "content": assistant_reply}
            )
            st.rerun()

        except Exception as e:
            st.error(f"Error during LangChain execution: {e}")
