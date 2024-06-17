from langchain_huggingface import HuggingFaceEndpoint

import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("HF_TOKEN")

repo_id="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=key)

print(llm.invoke("I am a bad boy"))