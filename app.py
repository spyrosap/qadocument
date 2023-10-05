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
from langchain.prompts import PromptTemplate

#load env variables 
load_dotenv(find_dotenv())

#load content (url or pdf)

def load_content(url) : 
    loader = loader = WebBaseLoader(url)
    data = loader.load()
    return data 

def load_pdf(file) :
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    #print(pages[0])
    return pages

#load juste one example 

pdf_content = load_pdf('Florida-Standard-Residential-Lease-Agreement.pdf')

#Split the content in chunks to be embedded into vectors using OpenAI embedding

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(pdf_content)

#create vectorstore
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

#create the prompt to answer the question in french

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Answer in French:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

#create the function to answer a question based on our vectorstore 

def answer(question) : 
    docs = vectorstore.similarity_search(question)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever(),chain_type_kwargs=chain_type_kwargs)
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
