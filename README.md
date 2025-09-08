# YouTube Chatbot

A conversational AI assistant that answers questions about YouTube videos using their transcripts. Built with LangChain, Google Generative AI, FAISS, and Streamlit.

## Features
- Fetches YouTube video transcripts automatically
- Splits transcripts into chunks for efficient retrieval
- Uses Google Generative AI for answering questions
- Interactive Q&A via command line and Streamlit web app

## Requirements
- Conda (recommended)
- Python 3.11
- Google API Key (for Generative AI)

## Setup
1. **Clone the repository**
   ```sh
   git clone <your-repo-url>
   cd YouTube_Chatbot
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
   - (Optional) Add other keys if needed for future features.

## Usage
### Command Line
Run the chatbot interactively:
```sh
python chatbot.py
```
Ask questions about the video transcript. Type 'quit' or 'exit' to stop.

### Streamlit Web App
Launch the web interface:
```sh
streamlit run app.py
```
- Enter a YouTube video URL or ID in the sidebar
- Click "Process Video" to fetch and process the transcript
- Ask questions in the chat interface

## File Structure
- `chatbot.py` — Command line Q&A chatbot
- `app.py` — Streamlit web app for interactive Q&A
- `environment.yml` — Conda environment specification
- `.env` — API keys and secrets (not tracked in git)

## API Keys
- **GOOGLE_API_KEY**: Required for Google Generative AI
- Other keys (OpenAI, HuggingFace, Gemini) are not required for current features
