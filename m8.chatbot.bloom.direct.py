import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import uuid

model_name = "bigscience/bloom-560m"


@st.cache_resource
def load_model_and_pipeline():
    try:
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
        return text_pipeline, tokenizer
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        st.stop()

text_pipeline, tokenizer = load_model_and_pipeline()


bloom560m_PROMPT = (
    "Generate creative, thought-provoking conversations, or ideas that encourage"
    "discussion and exploration. make prompts short, engaging, for conversation."
)
USER_PREFIX = "\n\nUser: "
AI_PREFIX = "\n\nAssistant: "

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

if "history_context" not in st.session_state:
    st.session_state.history_context = ""

def build_prompt_with_history(user_question: str) -> str:
    prompt_parts = [
        bloom560m_PROMPT,
        st.session_state.history_context,
        USER_PREFIX,
        user_question,
        AI_PREFIX 
    ]
    return "".join(prompt_parts)

def update_history_context(user_question: str, assistant_reply: str):
    new_turn = (
        USER_PREFIX + user_question +
        AI_PREFIX + assistant_reply
    )
    st.session_state.history_context += new_turn

st.set_page_config(page_title="Python chatbot", page_icon="📎")

st.markdown(
    """
    <div style="background-color:#f1f5c4; padding: 15px; border-radius: 10px;">
        <h1 style="color:#171714; text-align: center;"> Hugging Face BLOOM Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Direct communication with Hugging Face and the **Bloom** open-source model."
    "The chatBot has basic memory and will remember the conversation."
)

st.write("---")

# Display conversation messages
for msg in st.session_state.display_messages:
    role = "👤 user" if msg["role"] == "user" else "📎 Bloom_AI"
    
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
    placeholder="Example: write a story about dick?",
    key="user_input_area"
)

col1, col2 = st.columns(2)
with col1:
    send_clicked = st.button("Ask the bot")
with col2:
    clear_clicked = st.button("Clear conversation")


if clear_clicked:
    st.session_state.display_messages = []
    st.session_state.history_context = ""
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

if send_clicked and user_input.strip():
    st.session_state.display_messages.append({"role": "user", "content": user_input})

    with st.spinner("Bigscience/560m is thinking..."):
        try:
        
            full_prompt = build_prompt_with_history(user_input)

            response_list = text_pipeline(full_prompt)
            assistant_reply = response_list[0]['generated_text'].strip()

            if USER_PREFIX in assistant_reply:
                assistant_reply = assistant_reply.split(USER_PREFIX)[0].strip()
                
            update_history_context(user_input, assistant_reply)

            st.session_state.display_messages.append(
                {"role": "assistant", "content": assistant_reply}
            )
            st.rerun()

        except Exception as e:
            st.error(f"Error during text generation: {e}")
            st.session_state.display_messages.pop()
            st.rerun()
