import os
import openai
import pinecone
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

initial_setup = True

embedding = OpenAIEmbeddings()
# Initialize Pinecone index
index_name = os.environ["PINECONE_INDEX_NAME"]
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT"],
)
index = pinecone.Index(index_name=index_name)
vectordb = Pinecone(index=index, embedding=embedding, text_key="text")

if initial_setup:
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

    splits = text_splitter.split_documents(docs)

    # Adding PDFs to DB
    vectordb.add_documents(splits)  # only need to run this once!

    print("Documents successfully added to Pinecone!")

# Initialize the Assistant
prompt_template = """
    1. You're an assistant that answers informations only about Dracula and Frankenstein. Don't answer questions about cookies or anything else.

    {context}

    The queries are delimited by ####

    ####{question}####
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": prompt}

# Initialize OpenAI
llm_name = "gpt-3.5-turbo-0301"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
)

print("OpenAI and Pinecone successfully initialized!")

while True:
    question = input("\nPrompt: ")

    if question == "quit" or question == "exit":
        print("Exitting the prompt as requested.")
        break

    result = qa_chain(question)

    print(result["result"])

    docs = vectordb.similarity_search(question, k=3)

    source_document = result["source_documents"][0]
    print(source_document.metadata)
