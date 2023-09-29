import os
import openai
import pinecone
import sys
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

loader = PyPDFDirectoryLoader("data/")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], 
                environment=os.environ["PINECONE_ENVIRONMENT"])

print("OpenAI and Pinecone successfully initialized!")