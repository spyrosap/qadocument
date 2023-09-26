#import all deps

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import find_dotenv, load_dotenv
from langchain.retrievers import SVMRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


#load env variables 
load_dotenv(find_dotenv())

#search for the right article to read 

def load_content(url) : 
    loader = loader = WebBaseLoader(url)
    data = loader.load()
    return data 

data = load_content('https://www.growthunhinged.com/p/pleos-story-of-80-yoy-growth-on-the')


#Split the content in chunks to be embedded into vectors

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

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



answer("How did Pleo grow ?")






#print(data)
