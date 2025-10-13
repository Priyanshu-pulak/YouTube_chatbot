# YouTube Chatbot

A conversational AI assistant that answers questions about YouTube videos using their transcripts. Built with LangChain, Google Generative AI, FAISS, and Streamlit.

## Features
- Fetches YouTube video transcripts automatically
- Splits transcripts into chunks for efficient retrieval
- Uses Google Generative AI for answering questions

## Requirements
- Conda (recommended)
- Python 3.11
- Google API Key (for Generative AI)

## Setup
1. **Clone the repository**
   ```sh
   git clone https://github.com/Priyanshu-pulak/YouTube_chatbot.git
   cd YouTube_chatbot
   ```
2. **Create the environment**
   ```sh
   conda env create -f environment.yml
   conda activate youtube_chatbot
   ```
3. **Configure API Keys**
   - Create a `.env` file in the project root with your API keys:
     ```env
     GOOGLE_API_KEY="your-google-api-key"
     ```

## Usage
### Command Line
Run the chatbot interactively:
```sh
python chatbot.py
```
Ask questions about the video transcript. Type 'quit' or 'exit' to stop.

## File Structure
- `youtube_chatbot.py` — Command line Q&A chatbot
- `environment.yml` — Conda environment specification
- `.env` — API keys and secrets (not tracked in git)

## API Keys
- **GOOGLE_API_KEY**: Required for Google Generative AI