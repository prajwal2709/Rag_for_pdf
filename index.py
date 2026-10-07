import os
from langchain_core.vectorstores import InMemoryVectorStore 
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pdf_extract import paragraph_extract


load_dotenv()

embeddings=OpenAIEmbeddings(model="text-embedding-3-large",
                           api_key=os.getenv("OPENROUTER_API_KEY"),
                            base_url="https://openrouter.ai/api/v1"
                           )
                            

vector_store=InMemoryVectorStore(embeddings)

docs=paragraph_extract("pdfs/penguins_ACL.pdf")

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
texts=text_splitter.create_documents(docs)

print(f"split documentation into {len(texts)} chunks")

vector_store.add_documents(texts)

print(f"indexed {len(texts)} chunks into vector store")