import os
import sys

import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found. Please check your .env file.")
    sys.exit(1)

genai.configure(api_key=api_key)
print("apikey:",api_key)
model = genai.GenerativeModel(
    'gemini-2.0-flash',
    generation_config=genai.types.GenerationConfig(
       temperature=0.8,                          
    )
)


def load_database():
    # Use Google Embedding model
    embedding_function = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key,
    task_type="SEMANTIC_SIMILARITY" 
)
    # Load the database
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        print("Database not found. Please run 'create_database.py' first.")
        print("Command: python create_database.py")
        sys.exit(1)
    
    vectordb = Chroma(
        persist_directory=db_path, 
        embedding_function=embedding_function
    )
    
    return vectordb

def get_answer_from_doc(doc,quest):
    _find = doc.find(quest)
    _find_anwser = doc.find("**Soru",_find)
    if (_find < 0):
        return ""
    _find_start = doc.find("**Cevap",_find)
    if (_find_anwser < 0):
        _find_anwser = len(doc)
    return doc[_find_start:_find_anwser].replace("**Cevap:**","Cevap:").replace("\n","")
def get_answer(file,query, vectordb, top_k=3):
    with open(file, "r", encoding="utf-8") as f:
        md_content = f.read()
    retrieval_results = vectordb.similarity_search(query, k=top_k)
    context = "\n\n".join(["Soru:" + str(doc.page_content) + get_answer_from_doc(md_content,str(doc.page_content)) for doc in retrieval_results])
    # print("-------",context,"------")
    prompt = f"""
    
Sen Global AI Hub'ın akıl küpü chatbotusun. 
Aşağıdaki bağlam bilgisini kullanarak kullanıcının sorusuna net, doğru ve yardımsever bir şekilde yanıt ver. 
Eğer cevabı bağlam bilgisinde bulamıyorsan, bunu dürüstçe belirt ve bootcamp hakkında genel bilgi vermeye çalış.


BAĞLAM BİLGİSİ:
{context}

KULLANICI SORUSU: 
{query}

YANITINIZ:"""

    response = model.generate_content(prompt)

    return response.text

def main():
    print("=" * 50)
    print("Global AI Hub Chatbot'una Hoş Geldiniz!")
    print("Bootcamp hakkında sorularınızı sorabilirsiniz.")
    print("Çıkmak için 'q' veya 'çıkış' yazın.")
    print("=" * 50)

    vectordb = load_database()

    while True:
        query = input("\nSorunuz: ")
        if query.lower() in ["q", "çıkış", "exit", "quit"]:
            print("Global AI Hub Chatbot'undan çıkılıyor. İyi günler!")
            break
        try:
            answer = get_answer("data/soru_cevap.md",query, vectordb,top_k=7)
            print("\nCevap:", answer)
        except Exception as e:
            print(f"Bir hata oluştu: {e}")


if __name__ == "__main__":
    main()