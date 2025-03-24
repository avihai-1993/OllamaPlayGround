import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Streamlit Ollama Chatbot",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    ip = "localhost"
    port = "11434"
    url_gen = f"http://{ip}:{port}/api/generate"
    url_list = f"http://{ip}:{port}/api/tags"

    st.subheader("Streamlit Ollama Chatbot")

    try:
        res = requests.get(url_list)
        res.raise_for_status()
        available_models = [mod["model"] for mod in res.json().get("models", [])]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching models: {e}")
        available_models = []

    if available_models:
        selected_model = st.selectbox(
            "Pick a model available locally on your system ?", available_models
        )
    else:
        st.warning("You have not pulled any model from Ollama yet! ??")
        st.markdown(
            "[Pull model(s) from Ollama](https://ollama.com/library) ??",
            unsafe_allow_html=True,
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        #avatar = "??" if message["role"] == "assistant" else "??"
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter a prompt here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Give it a moment..."):
                try:
                    data_prompt = {"model": selected_model, "prompt": prompt}
                    response = requests.post(url_gen, json=data_prompt, stream=True)
                    response.raise_for_status()

                    fulltext = ""
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8")
                            result = json.loads(decoded_line)
                            generated_text = result.get("response", "")
                            fulltext += generated_text

                    st.markdown(fulltext)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": fulltext}
                    )

                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")
                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse response: {e}")


if __name__ == "__main__":
    main()
