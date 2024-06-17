import streamlit as st
import requests
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("HF_TOKEN")

# Function to perform text classification using Hugging Face's Inference API
def classify_text(text):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    inputs = tokenizer(text = text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]


def text_generation(text):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=key)
    return llm.invoke(text)


# Function to perform named entity recognition (NER) using Hugging Face's API
def recognize_entities(text):
    endpoint = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
    headers = {"Authorization": f"Bearer {key}"}  # Replace with your Hugging Face API token
    data = {"inputs": text}
    response = requests.post(endpoint, headers=headers, json=data)
    return response.json()


# Function to perform text summarization using Hugging Face's API
def summarize_text(text):
    endpoint = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {key}"}  # Replace with your Hugging Face API token
    data = {"inputs": text}
    response = requests.post(endpoint, headers=headers, json=data)
    return response.json()[0]["summary_text"]


# Function to perform question answering using Hugging Face's API
def answer_question(context, question):
    endpoint = "https://api-inference.huggingface.co/models/deepset/bert-base-cased-squad2"
    headers = {"Authorization": f"Bearer {key}"}  # Replace with your Hugging Face API token
    data = {"inputs": {"context": context, "question": question}}
    response = requests.post(endpoint, headers=headers, json=data)
    return response.json()["answer"]


st.title("NLP Tasks with Hugging Face Endpoints")

# Sidebar for navigation
st.sidebar.title("Choose NLP Task")
task = st.sidebar.selectbox("Task", ["Text Classification", "Named Entity Recognition", "Text-Generation", "Text Summarization", "Question Answering"])

if task == "Text Classification":
    st.header("Text Classification")
    text = st.text_area("Enter text for classification")
    if st.button("Classify"):
        if text:
            result = classify_text(text)
            st.write(result)
        else:
            st.write("Please enter text to classify")

elif task == "Named Entity Recognition":
    st.header("Named Entity Recognition")
    text = st.text_area("Enter text for NER")
    if st.button("Recognize Entities"):
        if text:
            result = recognize_entities(text)
            st.write(result)
        else:
            st.write("Please enter text for NER")

elif task == "Text-Generation":
    st.header("Text-Generation")
    text = st.text_area("Enter text for Text-Generation")
    if st.button("Generate text"):
        if text:
            result = text_generation(text)
            st.write(result)
        else:
            st.write("Please enter text for text generation")

elif task == "Text Summarization":
    st.header("Text Summarization")
    text = st.text_area("Enter text for summarization")
    if st.button("Summarize"):
        if text:
            result = summarize_text(text)
            st.write(result)
        else:
            st.write("Please enter text to summarize")

elif task == "Question Answering":
    st.header("Question Answering")
    context = st.text_area("Enter context")
    question = st.text_input("Enter question")
    if st.button("Get Answer"):
        if context and question:
            result = answer_question(context, question)
            st.write(result)
        else:
            st.write("Please enter both context and question")
