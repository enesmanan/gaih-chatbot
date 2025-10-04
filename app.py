import os
import sys
import uuid
from datetime import datetime

import google.generativeai as genai
import markdown
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from create_database import create_database

# ----- Configuration and Setup -----
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found. Please check your .env file.")
    sys.exit(1)

genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    "gemini-2.0-flash",
    generation_config=genai.types.GenerationConfig(
        temperature=0.57
        #    max_output_tokens=2048,
        #    top_p=0.9,
        #    top_k=40,
    ),
)

# ----- Database Functions -----
def load_database():
    # Use Google Embedding model
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=api_key
    )

    # Load the database
    db_path = "./chroma_db"
    collection_name = "gaih-chatbot"
    client_settings = Settings(anonymized_telemetry=False)
    if not os.path.exists(db_path):
        try:
            print("Database not found. Creating now...")
            create_database()
        except Exception as e:
            print(f"Failed to create database: {e}")
            sys.exit(1)

    try:
        vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_function,
            collection_name=collection_name,
            client_settings=client_settings,
        )
        return vectordb
    except Exception as e:
        print(f"Failed to load vector DB ({e}). Rebuilding...")
        try:
            create_database()
            vectordb = Chroma(
                persist_directory=db_path,
                embedding_function=embedding_function,
                collection_name=collection_name,
                client_settings=client_settings,
            )
            return vectordb
        except Exception as e2:
            print(f"Rebuild failed: {e2}")
            sys.exit(1)

# ----- RAG & AI Functions -----
def get_answer(query, vectordb, top_k=3):
    # Get similar documents from vector database
    retrieval_results = vectordb.similarity_search(query, k=top_k)
    context = "\n\n".join([doc.page_content for doc in retrieval_results])

    # Prepare prompt to send to Gemini
    prompt = f"""
    
Sen Global AI Hub'ın akıl küpü chatbotusun. 
Aşağıdaki bağlam bilgisini kullanarak kullanıcının sorusuna net, doğru ve yardımsever bir şekilde yanıt ver. 
Eğer cevabı bağlam bilgisinde bulamıyorsan, bunu dürüstçe belirt ve Global AI Hub bootcamp'leri hakkında genel bilgi vermeye çalış.
Bağlam içinde olmayan Python, Javascript veya herhangi bir programlama dili kodlarını gösterme veya açıklama. Bunun yerine gene Global AI Hub bootcamp'leri hakkında genel bilgi vermeye çalış.

BAĞLAM BİLGİSİ:
{context}

KULLANICI SORUSU: 
{query}

YANITINIZ:"""

    # Get response from Gemini model
    response = model.generate_content(prompt)

    return response.text

# ----- Helper Functions -----
# Markdown processing function
def render_markdown(text):
    # Convert Markdown format to HTML
    html = markdown.markdown(
        text,
        extensions=[
            "markdown.extensions.extra",
            "markdown.extensions.codehilite",
            "markdown.extensions.smarty",
            "markdown.extensions.nl2br",
            "markdown.extensions.sane_lists",
        ],
    )
    return html

# ----- Flask Application Setup -----
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session

# Initialize vector database
vectordb = load_database()

# Dictionary to store session information
conversations = {}

# ----- Flask Routes -----
@app.route("/")
def index():
    # Check user session ID, create a new session if not exists
    if "session_id" not in session:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id
        conversations[session_id] = {
            "id": session_id,
            "title": f"New Conversation {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            "created_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
            "messages": [],
        }

    # Get current session information
    session_id = session["session_id"]
    current_conversation = conversations.get(
        session_id,
        {
            "id": session_id,
            "title": f"New Conversation {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            "created_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
            "messages": [],
        },
    )

    # List all sessions
    all_conversations = [conv for conv in conversations.values()]

    return render_template(
        "index.html",
        conversation_history=current_conversation.get("messages", []),
        conversations=all_conversations,
        current_session=session_id,
        renderMarkdown=render_markdown,
    )  # Markdown processing function


@app.route("/send_message", methods=["POST"])
def send_message():
    data = request.json
    user_message = data.get("message", "")

    # Save user message
    session_id = session.get("session_id")
    if session_id not in conversations:
        conversations[session_id] = {
            "id": session_id,
            "title": f"Conversation {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            "created_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
            "messages": [],
        }

    # Save messages
    conversations[session_id]["messages"].append(
        {"role": "user", "content": user_message}
    )

    # Set first message as conversation title
    if len(conversations[session_id]["messages"]) == 1:
        # Create title from user's first message (max 30 characters)
        title = user_message[:30] + "..." if len(user_message) > 30 else user_message
        conversations[session_id]["title"] = title

    # Get answer
    try:
        bot_response = get_answer(user_message, vectordb, top_k=10)

        # Save bot response
        conversations[session_id]["messages"].append(
            {"role": "bot", "content": bot_response}
        )

        # Return all conversations as JSON
        all_conversations = [conv for conv in conversations.values()]

        return jsonify({"response": bot_response, "conversations": all_conversations})

    except Exception as e:
        return jsonify(
            {
                "response": f"Sorry, an error occurred: {str(e)}",
                "conversations": [conv for conv in conversations.values()],
            }
        )


@app.route("/new_chat", methods=["POST"])
def new_chat():
    # Create new session
    session_id = str(uuid.uuid4())
    session["session_id"] = session_id
    conversations[session_id] = {
        "id": session_id,
        "title": f"New Conversation {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        "created_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "messages": [],
    }

    return jsonify({"success": True})


@app.route("/conversation/<session_id>")
def load_conversation(session_id):
    if session_id in conversations:
        session["session_id"] = session_id
        return redirect(url_for("index"))
    else:
        return redirect(url_for("index"))


# ----- Application Entry Point -----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
