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

# make setup happen based on conditional (setup is lines 17-30)
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

embedding = OpenAIEmbeddings()
# Initialize Pinecone index
indexName = os.environ["PINECONE_INDEX_NAME"]
index = pinecone.Index(index_name=indexName)
vectordb = Pinecone(index=index, embedding=embedding, text_key="text")

vectordb.add_documents(splits) # only need to run this once!

# put prompt - print response in a while loop (lines 41-end of file)
question = input("\nPrompt: ")
docs = vectordb.similarity_search(question,k=3)

llm_name = "gpt-3.5-turbo-0301"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

result = qa_chain({"query": question})

print(result["result"])

source_document = result["source_documents"][0]
print(source_document.metadata)

# add a system message
# add some instructions for the bot to follow (i.e. you're a bot about dracula and frankenstein - don't answer questions about cookies
# add a way to exit the loop (i.e. type "exit" or "quit")
