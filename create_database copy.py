import os
import re
import chromadb
import google.generativeai as genai
import markdown
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
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
    api_key = "AIzaSyA5tm44ev1KhCDYQdPS7rL3mw7kqJsheHw"

    if not api_key:
        print("Error: GOOGLE_API_KEY not found. Please check your .env file.")
        return

    genai.configure(api_key=api_key)

    # Read markdown data file
    with open("data/soru_cevap.md", "r", encoding="utf-8") as f:
        md_content = f.read()
    
    pattern = r"\*\*Soru:\*\*(.*?)\*\*Cevap:\*\*"
    matches = re.findall(pattern, md_content, re.DOTALL)

    chunks = [str(match.strip()).replace("\n","") for match in matches]


    # Clean data from markdown format



    # Split text into chunks
    documents = [Document(page_content=chunk) for chunk in chunks]

    print(f"{len(documents)} document chunks created.")

    # Use Google Embedding model
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
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
