"""Luganda chat with Kintu 2. Run: streamlit run streamlit_app.py"""

import time

import streamlit as st

MODELS = {
    "Kintu-2.0-Instruct": "henryhale/Kintu-2.0-30M-Instruct",
    "Kintu-2.0": "henryhale/Kintu-2.0-30M",
}
SYSTEM = "Oli muyambi ow'amagezi ayamba abantu mu Luganda."
SUGGESTIONS = [
    "Oli otya mukwano?",
    "Njagala kufumba emmere.",
    "Mwana wange tayagala kusoma. Nkole ntya?",
    "Nnyinza ntya okutereka ssente okuva mu musaala omutono?",
]

GENERATION = dict(max_new_tokens=128, do_sample=True, temperature=0.5, top_p=0.8, repetition_penalty=1.1)
CONTEXT = 512  # the models were trained on at most 512 tokens

st.set_page_config(page_title="Kintu-2.0", page_icon="💬")
st.title("💬 Kintu")
st.caption("Chat in Luganda with a small 30M-parameter model. Replies may not be accurate.")


@st.cache_resource(show_spinner="Loading Kintu…")  # once per model per server process
def load(repo):
    start = time.time()
    print(f"loading {repo}…", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok, model = AutoTokenizer.from_pretrained(repo), AutoModelForCausalLM.from_pretrained(repo)
    print(f"model loaded in {time.time() - start:.0f} s", flush=True)
    return tok, model


messages = st.session_state.setdefault("messages", [])


def reply(tok, model, history):
    # Drop the oldest exchanges until the chat plus the reply fits the context.
    while True:
        enc = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}] + history,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if enc["input_ids"].shape[1] + GENERATION["max_new_tokens"] <= CONTEXT or len(history) <= 1:
            break
        history = history[2:]
    enc.pop("token_type_ids", None)  # generate() rejects it
    out = model.generate(**enc, **GENERATION, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def send(text):
    st.session_state.messages.append({"role": "user", "content": text})


with st.sidebar:
    st.title("Settings")
    name = st.selectbox("Model", list(MODELS), help="Instruct is Kintu-2.0 fine-tuned further on chats and preferences.")
    st.button("New chat", on_click=messages.clear, disabled=not messages)

tok, model = load(MODELS[name])

if not messages:
    st.write("Try one:")
    for col, s in zip(st.columns(2) * 2, SUGGESTIONS):
        col.button(s, on_click=send, args=(s,), width="stretch")

for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if messages and messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            text = reply(tok, model, messages)
        text = text or "_(empty reply, try again)_"
        st.markdown(text)
    messages.append({"role": "assistant", "content": text})

st.chat_input("Write in Luganda…", key="draft", max_chars=400, on_submit=lambda: send(st.session_state.draft))
