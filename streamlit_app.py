"""Luganda chat with Kintu. Run: streamlit run app.py"""

import time

import streamlit as st

REPO = "henryhale/Kintu-1.0-39M"
SUGGESTIONS = ["Oli otya?", "Wasuze otya nno?", "Osiibye otyanno ssebo?"]

st.set_page_config(page_title="Kintu-1.0", page_icon="💬")
st.title("💬 Kintu")
st.caption("Chat in Luganda with a small 39M-parameter model. Replies can be short or off-topic.")


@st.cache_resource(show_spinner="Loading Kintu-1.0…")  # once per server process
def load():
    start = time.time()
    print("loading model…", flush=True)
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    tok, model = PreTrainedTokenizerFast.from_pretrained(REPO), AutoModelForCausalLM.from_pretrained(REPO)
    print(f"model loaded in {time.time() - start:.0f} s", flush=True)
    return tok, model


tok, model = load()
messages = st.session_state.setdefault("messages", [])


def reply(message, temperature, top_p, max_new_tokens):
    inputs = tok(f"[CLS]Omuntu: {message} Omuyambi:", return_tensors="pt", add_special_tokens=False)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tok.convert_tokens_to_ids("[SEP]"),
        pad_token_id=tok.convert_tokens_to_ids("<pad>"),
    )
    text = tok.decode(out[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
    return text.split("Omuntu:")[0].strip()  # cut it off if it starts writing the user's next turn


def send(text):
    st.session_state.messages.append({"role": "user", "content": text})


with st.sidebar:
    st.title("Settings")
    temperature = st.slider("Temperature", 0.1, 1.5, 1.0, 0.05, help="Lower = safer but more repetitive replies.")
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05, help="Sample only from the likeliest words covering this share.")
    max_new_tokens = st.slider("Max reply length (tokens)", 10, 120, 40, 10)
    st.button("New chat", on_click=messages.clear, disabled=not messages)

if not messages:
    st.write("Try one:")
    for col, s in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
        col.button(s, on_click=send, args=(s,))

for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if messages and messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            text = reply(messages[-1]["content"], temperature, top_p, max_new_tokens)
        text = text or "_(empty reply, try again)_"
        st.markdown(text)
    messages.append({"role": "assistant", "content": text})

st.chat_input("Write in Luganda…", key="draft", max_chars=400, on_submit=lambda: send(st.session_state.draft))