import streamlit as st
from langchain_classic.chains.conversation.base import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
import torch
from transformers import BloomTokenizerFast, AutoModelForCausalLM, pipeline


# Loading (Hugging Face & LangChain)
@st.cache_resource
def load_llm_and_chain():

    model_name = "bloom-560m"

    # tokenizer and ai model
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True #true=trusted, false= not trusted.  which one is better?
    )

    # create Text  Pipeline
    text_pipeline = pipeline(
        "text-generation", #warning, dont change this text.
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512, # ai response limit
        temperature=0.7,    #controls randomness
        do_sample=True,
        top_k=50, #top 50 most probable tokens
        top_p=0.95, #removes bottom 5% and only uses upper 95%
        return_full_text=False  # doesnt return prompt
    )

    # pipeline
    llm = HuggingFacePipeline(pipeline=text_pipeline)

    memory = ConversationBufferMemory(memory_key="history")

    # prompt layout with history variable for memory.
    template = """Explain concepts clearly, use simple examples, keep responses brief.

        Conversation:
        {history}
        Human: {input}
        AI:"""

    layout = PromptTemplate(input_variables=["history", "input"], template=template)

    # langChain conversation chain (LLM + Prompt + Memory)
    conversation = ConversationChain(
        prompt=layout,
        llm=llm,
        memory=memory,
        verbose=False  # True shows prompt sent to llm
    )

    return conversation


chain = load_llm_and_chain()

# page layout and html
st.set_page_config(page_title="Python chatbot", page_icon="")

st.markdown(
    """
    <div style="background-color:#f1f5c4; padding: 15px; border-radius: 10px;">
        <h1 style="color:white; text-align: center;"> LangChain + Hugging Face Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True #true vs false??
)

st.write(
    "Hugging Face and the **Bloom** open-source model.\n"
    "chatBot has memory and will remembers the conversation.\n"  #do i need the \n?
)

st.write("---")

# display conversation
if "messages" not in st.session_state:
    st.session_state.display_messages = []

for msg in st.session_state.display_messages:
    role = "👤 student" if msg["role"] == "user" else "📎 Bloom_AI"

    # html for display
    color = "#E0F7FA"
    st.markdown(
        f"""
        <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
            <b>{role}:</b> {msg['content']}
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("---")

# user input
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

# clear chat
if clear_clicked:
    chain.memory.clear()
    st.session_state.display_messages = []
    st.rerun()


if send_clicked and user_input.strip():
    st.session_state.display_messages.append({"role": "user", "content": user_input})

    with st.spinner("Bloom_AI is thinking..."):
        try:
            response = chain.invoke(user_input)
            assistant_reply = response['response']

            st.session_state.display_messages.append(
                {"role": "assistant", "content": assistant_reply}
            )
            st.rerun()

        except Exception as e:
            st.error(f"Error during LangChain execution: {e}")