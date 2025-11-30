import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import uuid

model_name = "bigscience/bloom-560m"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

text_pipeline = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    return_full_text=False)

llm = HuggingFacePipeline()

user_template = f"{{question}}"

prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "explain concepts clearly, use simple examples, keep responses brief."),
            MessagesPlaceholder(variable_name="history"),
            ("human", user_template),
        ]
    )

core_chain = prompt | model

store = {}

def get_session_id(session_id: str):
    if session_id not in store:
        store[session_id] = uuid.uuid4().hex
    return store[session_id]

chain_history = RunnableWithMessageHistory(
    core_chain,
    get_session_id,
    input_messages_key="question",
    history_messages_key="history",
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.display_messages = []

# page layout and html
st.set_page_config(page_title="Python chatbot", page_icon="📎")

st.markdown(
    """
    <div style="background-color:#f1f5c4; padding: 15px; border-radius: 10px;">
        <h1 style="color:#171714; text-align: center;"> LangChain + Hugging Face Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Hugging Face and the **Bloom** open-source model."
    "chatBot has memory and will remember the conversation."
)

st.write("---")

for msg in st.session_state.display_messages:
    role = "👤 student" if msg["role"] == "user" else "📎 Bloom_AI"
    color = "#f1f5c4"

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

    with st.spinner("Bloom_AI is thinking..."):
        try:
            response = core_chain.invoke(
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