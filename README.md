# Global AI Hub Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot system developed for Global AI Hub using Google's Gemini 2.0 Flash model and ChromaDB. The system efficiently retrieves and answers questions about Global AI Hub bootcamps.


## Configuration

GAIH Asistan uses Google Gemini API with the following configuration:

- **Model**: `gemini-2.0-flash`
- **Embedding Model**: `models/embedding-001`
- **Top-k Retrieval**: Retrieves 10 most relevant documents by default
- **Generative Model Parameters**: temperature=0.57
- **Chunking Strategy**: 2000 character chunks with 300 character overlap



## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/enesmanan/gaih-chatbot.git
   cd gaih-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file and add your Google API key: 
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. Create the vector database:
   ```bash
   python create_database.py
   ```

5. Run the chatbot:
   ```bash
   python app.py
   ```


## Project Structure

```
gaih-chatbot/
├── data/              
│   └── soru_cevap.md  # Q&A dataset in markdown format
├── app.py             # Main application script
├── create_database.py # Database creation script
├── chroma_db/         # ChromaDB vector database directory
├── static/            # CSS and static assets
├── templates/         # HTML templates
├── requirements.txt   # Project dependencies
├── .env               # Environment variables configuration
└── README.md          # Project documentation
```