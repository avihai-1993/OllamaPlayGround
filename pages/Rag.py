import streamlit as st

from util import *

VECTOR_DB = InMemoryVectorDB()

st.title("📝 File Q&A RAG Custom")

#uploaded_file = st.file_uploader("Upload an article", type=("txt", "md","pdf" ,"json"))

#question = st.text_input(
#    "Ask something about the article",
#    placeholder="Can you give me a short summary?",
    #disabled=not uploaded_file,
#)


#if uploaded_file and question:
#    article = uploaded_file.read().decode()
#    st.write(article)

PATH_TO_DOCS = "C:\\Users\\AvihaiMaslawi(IS-TA)\\PycharmProjects\\OllamaWithStreamlit\\docs"

if not VECTOR_DB.db: #check if the db is empty
    with st.spinner("Loading and processing documents..."):
        documents = load_documents(PATH_TO_DOCS)
        chunks = chunk_documents(documents)
        for chunk in chunks:
            embedding = ollama_embedding(chunk["content"])
            VECTOR_DB.add(embedding, chunk["content"], chunk["source"])

query = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    #disabled=not uploaded_file,
)
#print(VECTOR_DB)
if query:
    with st.spinner("Generating response..."):
        query_embedding = ollama_embedding(query)
        #print(query_embedding)
        relevant_chunks = VECTOR_DB.retrieve(query_embedding)
        context_text = "\n\n".join([chunk["content"] for chunk in relevant_chunks])

        prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
        response, _ = ollama_generate(prompt)

        st.write("Response:")
        st.write(response)
        st.write("Relevant Documents:")
        for chunk in relevant_chunks:
            st.write(f"- {chunk['source']}")
