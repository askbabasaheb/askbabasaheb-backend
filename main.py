from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

app = FastAPI()

class Query(BaseModel):
    question: str

documents = []
for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(f"docs/{file}")
        documents.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

@app.post("/ask")
async def ask(q: Query):
    result = qa.run(q.question)
    return {"answer": result}
