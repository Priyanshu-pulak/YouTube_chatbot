# YouTube Chatbot (Modular)

A conversational AI assistant that fetches YouTube video transcripts, builds vector stores for efficient retrieval, and uses LangChain + Google Generative AI to answer questions or generate summaries. The project follows a clean modular architecture with separation of concerns.

## Features

- ✅ Fetches YouTube video transcripts automatically
- ✅ Splits transcripts into chunks for efficient vector search
- ✅ Query classification (summary vs. question-answering)
- ✅ RAG-based Q&A using FAISS vector stores
- ✅ Intelligent summarization with context-aware retrieval
- ✅ Shows which chain (summary/qa) handled each query
- ✅ Modular codebase for easy testing and maintenance

## Requirements

- Python 3.11+
- Conda (recommended) or virtualenv
- Google API Key (for Generative AI)

## Project Structure

```
YouTube_chatbot/
├── .env                      # Environment variables (API keys)
├── .env.example              # Example environment file
├── .gitignore
├── environment.yml           # Conda environment spec
├── main.py                   # Main entrypoint
├── README.md
└── src/
    ├── __init__.py
    ├── youtube_chatbot.py    # Main orchestrator
    ├── utils.py              # Utility functions (fetch_transcript, split_text)
    ├── chain/
    │   ├── __init__.py
    │   ├── classification_chain.py   # Query classification logic
    │   ├── qa_chain.py               # Question-answering chain
    │   └── summary_chain.py          # Summarization chain
    ├── prompt_templates/
    │   ├── __init__.py
    │   └── prompt.py         # All prompt templates
    └── vector_stores/
        ├── __init__.py
        ├── qa_vector_store.py        # QA FAISS store builder
        └── summary_vector_store.py   # Summary FAISS store builder
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Priyanshu-pulak/YouTube_chatbot.git
cd YouTube_chatbot
```

### 2. Create the environment

Using conda (recommended):

```bash
conda env create -f environment.yml
conda activate youtube_chatbot
```

### 3. Configure API Keys

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY="your-google-api-key-here"
```

Get your Google API key from: https://makersuite.google.com/app/apikey

## Usage

### Command Line

Run the chatbot interactively:

```bash
python main.py
```

The chatbot will:

1. Fetch and process the default YouTube video transcript
2. Generate chunk summaries (with progress indicators)
3. Build QA and summary vector stores
4. Start an interactive Q&A session

Example interaction:

```
YouTube Video Q&A System
Ask questions about the video. Type 'quit' or 'exit' to stop.

Your question: 
give me a summary of the video

[handled by: summary]
The video discusses the main concepts of machine learning...

Your question:
what is gradient descent?

[handled by: qa]
Gradient descent is an optimization algorithm used to minimize...
```

Type `quit`, `exit`, or `q` to stop.

### Customizing the Video URL

Edit `src/youtube_chatbot.py` and change the default video URL in the `youtube_chatbot()` function, or modify `main.py` to accept a URL as a command-line argument.

## How It Works

1. **Transcript Fetching** (`src/utils.py`)

   - Extracts video ID from YouTube URL
   - Fetches transcript using `youtube-transcript-api`
2. **Text Processing** (`src/utils.py`)

   - Splits transcript into chunks (1000 chars, 200 overlap)
   - Creates LangChain Document objects
3. **Vector Stores** (`src/vector_stores/`)

   - Builds FAISS vector stores for QA and summaries
   - Uses Google's text-embedding-004 model
4. **Chain Building** (`src/chain/`)

   - **QA Chain**: Retrieves relevant chunks and answers questions
   - **Summary Chain**: Generates and retrieves summaries
   - **Classification Chain**: Routes queries to appropriate handler
5. **Query Processing** (`src/youtube_chatbot.py`)

   - Classifies user query (summary vs. question)
   - Routes to the correct chain
   - Returns answer with chain identifier

## Development

### Module Responsibilities

- `src/utils.py` — Transcript fetching and text splitting utilities
- `src/chain/qa_chain.py` — QA chain construction
- `src/chain/summary_chain.py` — Summary chain construction
- `src/chain/classification_chain.py` — Query classification logic
- `src/prompt_templates/prompt.py` — All prompt templates
- `src/vector_stores/` — FAISS vectorstore builders
- `src/youtube_chatbot.py` — Main orchestrator (thin, imports from modules)

### Running as a Package

```bash
# From project root
python main.py
```

## Troubleshooting

### ModuleNotFoundError

- Ensure you're running from the project root
- Use `python main.py` or `python -m src.youtube_chatbot`
- Check that `src/` has `__init__.py` files

### API Authentication Errors

- Verify your `.env` file contains `GOOGLE_API_KEY`
- Test API key: `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GOOGLE_API_KEY'))"`

### Transcript Disabled Error

- Some videos have transcripts disabled
- Try a different video URL
- Check video language (currently supports English)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Follow the modular structure
4. Add tests for new features
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## API Keys & Services

- **GOOGLE_API_KEY**: Required for Google Generative AI (gemini-2.5-flash-lite) and embeddings (text-embedding-004)
- Get your key: https://makersuite.google.com/app/apikey

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [Google Generative AI](https://ai.google.dev/)
- Vector search via [FAISS](https://github.com/facebookresearch/faiss)
