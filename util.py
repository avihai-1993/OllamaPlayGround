'''
if uploaded_file and question and not anthropic_api_key:
    st.info("Please add your Anthropic API key to continue.")

if uploaded_file and question and anthropic_api_key:
    article = uploaded_file.read().decode()
    #prompt = f"""{anthropic.HUMAN_PROMPT} Here's an article:\n\n{article}\n\n\n\n{question}{anthropic.AI_PROMPT}"""

    #client = anthropic.Client(api_key=anthropic_api_key)

      response = client.completions.create(
        prompt=prompt,
        #stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1", #"claude-2" for Claude 2 model
        max_tokens_to_sample=100,
    )

    st.write("### Answer")
    st.write(response.completion)

'''

import streamlit as st
import os
import requests
import json
from typing import List
import numpy as np
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

ip = "localhost"
port = "11434"
OLLAMA_API_BASE = f"http://{ip}:{port}/api"
EMBEDDING_MODEL_NAME = "nomic-embed-text:latest"
#EMBEDDING_MODEL_NAME = "mxbai-embed-large:latest"
LLM_MODEL_NAME = "llama3.2:3b"

class InMemoryVectorDB:
    def __init__(self):
        self.db = {}

    def add(self, embedding: List[float], content: str, source: str):
        index = len(self.db)
        self.db[index] = {"embedding": embedding, "content": content, "source": source}

    def retrieve(self, query_embedding: List[float], top_k: int = 4) -> List[dict]:
        similarities = []
        for index, data in self.db.items():
            similarity = cosine_similarity(np.array(query_embedding), np.array(data["embedding"]))[0][0]
            similarities.append((similarity, index))

        similarities.sort(reverse=True)
        relevant_chunks = [self.db[index] for _, index in similarities[:top_k]]
        return relevant_chunks

    def __str__(self):
        print(self.db)
        return "_____"

def ask_ollama(url_gen :str ,selected_model:str ,prompt:str ):
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


def ollama_embedding(text: str, model: str = EMBEDDING_MODEL_NAME) -> List[float]:
    """Generates embeddings using Ollama."""
    #url = f"{OLLAMA_API_BASE}/embeddings"
    url = f"{OLLAMA_API_BASE}/embed"
    payload = {"model": model, "input": text}
    #print("((((((((((((((((((((((((((((((((",url)
    response = requests.post(url, json=payload)
    response.raise_for_status()
    #print(response.json())
    return response.json()["embeddings"]

def ollama_generate(prompt: str, model: str = LLM_MODEL_NAME, context: List[int] = None) -> str:
    """Generates text using Ollama."""
    url = f"{OLLAMA_API_BASE}/generate"
    payload = {"model": model, "prompt": prompt, "context": context}
    response = requests.post(url, json=payload, stream=True)
    response.raise_for_status()

    full_response = ""
    for line in response.iter_lines():
        if line:
            json_line = json.loads(line)
            if "response" in json_line:
                full_response += json_line["response"]
            if "context" in json_line:
              context = json_line["context"]
    return full_response, context

def load_documents(docs_path: str) -> List[dict]:
    """Loads text from .txt and .pdf files."""
    documents = []
    for filename in os.listdir(docs_path):
        filepath = os.path.join(docs_path, filename)
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append({"content": content, "source": filename})
        elif filename.endswith(".pdf"):
            with open(filepath, "rb") as f:
                reader = PdfReader(f)
                content = "".join(page.extract_text() for page in reader.pages)
                documents.append({"content": content, "source": filename})
    return documents

def chunk_documents(documents: List[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[dict]:
    """Splits documents into smaller chunks."""
    chunks = []
    for doc in documents:
        content = doc["content"]
        source = doc["source"]
        for i in range(0, len(content), chunk_size - chunk_overlap):
            chunk = content[i:i + chunk_size]
            chunks.append({"content": chunk, "source": source})
    return chunks
