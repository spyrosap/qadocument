#import all deps

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import find_dotenv, load_dotenv
from langchain.retrievers import SVMRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from fastapi import FastAPI
from pydantic import BaseModel



#load env variables 
load_dotenv(find_dotenv())

#search for the right article to read 

def load_content(url) : 
    loader = loader = WebBaseLoader(url)
    data = loader.load()
    return data 

def load_pdf(file) :
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    #print(pages[0])
    return pages

pdf_content = load_pdf('Florida-Standard-Residential-Lease-Agreement.pdf')



#Split the content in chunks to be embedded into vectors

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(pdf_content)

#print(all_splits)

#create vectorstore
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

#create the function to answer a question based on our vectorstore 

def answer(question) : 
    docs = vectorstore.similarity_search(question)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())
    answer = qa_chain({"query": question})
    print(answer["result"])
    return answer



#Set the API endpoint to ask the question 

app = FastAPI()

class Query(BaseModel):
    query:str


@app.post('/')
def askDocument(query : Query):
    query = query.query 
    result = answer(query)
    actual_result = answer['output']
    return actual_result

#answer("What is this contract saying ?")




#print(data)
