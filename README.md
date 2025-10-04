# Global AI Hub Chatbot 

A production-ready, Retrieval-Augmented Generation (RAG) chatbot built for Global AI Hub. It indexes Q&A content, retrieves the most relevant chunks with Chroma, and generates helpful answers using Google Gemini.

### Features
- Simple RAG pipeline: chunk → embed → store → retrieve → generate
- Google Gemini 2.0 Flash for response generation
- `text-embedding-004` for high‑quality embeddings
- Flask UI, Markdown rendering, persistent Chroma vector store

### Tech Stack
- Backend: Flask
- RAG: LangChain, Chroma
- LLM: Google Gemini (`gemini-2.0-flash`)
- Embeddings: Google `models/text-embedding-004`
- Data: Markdown Q&A (`data/soru_cevap.md`)

## Requirements
- Python 3.10+
- A valid Google API key with access to Gemini models

## Setup
1. Clone the repository
   ```bash
   git clone https://github.com/enesmanan/gaih-chatbot.git
   cd gaih-chatbot
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment
   Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. Create the vector database (first run or when data/model changes)
   ```bash
   python create_database.py
   ```

5. Start the app
   ```bash
   python app.py
   ```
   Then open your browser at `http://localhost:5000`.

## Configuration (defaults)
- **Generative model**: `gemini-2.0-flash` (temperature=0.57)
- **Embedding model**: `models/text-embedding-004`
- **Chunking**: 2000 characters, 300 overlap
- **Retrieval k**: 10

## Project Structure
```
gaih-chatbot/
├── data/
│   └── soru_cevap.md         # Q&A dataset (Markdown)
├── app.py                    # Flask app + RAG query flow
├── create_database.py        # Chunk + embed + persist to Chroma
├── chroma_db/                # Persisted vector store (auto-created)
├── static/                   # CSS and images
├── templates/                # HTML templates
├── requirements.txt          # Dependencies
├── .env                      # GOOGLE_API_KEY (not committed)
└── README.md                 # This file
```