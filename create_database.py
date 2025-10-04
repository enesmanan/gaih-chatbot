import os

import chromadb
import google.generativeai as genai
import markdown
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def markdown_to_text(markdown_content):
    # Converting simple markdown formats to plain text
    # while preserving the question-answer format
    text = markdown_content
    # Clean headings and formatting marks
    text = text.replace("**Soru:**", "Soru: ")
    text = text.replace("**Cevap:**", "Cevap: ")
    return text


def create_database():
    print("Creating database...")
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("Error: GOOGLE_API_KEY not found. Please check your .env file.")
        return

    genai.configure(api_key=api_key)

    # Read markdown data file
    with open("data/soru_cevap.md", "r", encoding="utf-8") as f:
        md_content = f.read()

    # Clean data from markdown format
    text_content = markdown_to_text(md_content)

    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        length_function=len,
    )

    # Split text into chunks
    chunks = text_splitter.split_text(text_content)
    documents = [Document(page_content=chunk) for chunk in chunks]

    print(f"{len(documents)} document chunks created.")

    # Use Google Embedding model
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=api_key
    )

    # Create vector database
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory="./chroma_db",
    )

    print("Database successfully created and saved to 'chroma_db' folder.")
    return vectordb


if __name__ == "__main__":
    load_dotenv()
    create_database()
