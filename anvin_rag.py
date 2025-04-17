import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import json

class AnvinRAG:
    def __init__(self):
        # Initialize embeddings using Ollama's nomic-embed-text
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        # Initialize vector store
        self.vectorstore = None
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize Ollama
        self.llm = Ollama(
            model="anvin",
            base_url="http://localhost:11434"
        )
        
    def load_documents(self, file_path):
        """Load documents from a PDF file."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        texts = self.text_splitter.split_documents(documents)
        
        # Create or update vector store
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory="anvin_db"
            )
        else:
            self.vectorstore.add_documents(texts)
            
    def search(self, query, k=3):
        """Search for relevant documents."""
        if self.vectorstore is None:
            return "No documents loaded. Load some documents first."
        
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def query(self, question):
        """Query the RAG system."""
        if self.vectorstore is None:
            return "No documents loaded. Load some documents first."
        
        # Create retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True
        )
        
        # Get response
        result = qa_chain({"query": question})
        
        # Format response in ANVIN's style
        response = {
            "answer": result["result"],
            "sources": [doc.page_content for doc in result["source_documents"]]
        }
        
        return response

def main():
    # Initialize RAG system
    rag = AnvinRAG()
    
    # Load documents
    print("Loading documents...")
    rag.load_documents("Anvin_Shibu_resume.pdf")
    
    # Example query
    question = "What are Anvin's skills?"
    response = rag.query(question)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {response['answer']}")
    print("\nSources:")
    for i, source in enumerate(response['sources'], 1):
        print(f"{i}. {source[:200]}...")

if __name__ == "__main__":
    main() 