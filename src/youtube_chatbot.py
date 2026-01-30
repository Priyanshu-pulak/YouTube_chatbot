from src.chain import build_chatbot_chain
    
def youtube_chatbot(video_url: str = "https://www.youtube.com/watch?v=XmRrGzR6udg&list=PLgUwDviBIf0rAuz8tVcM0AymmhTRsfaLU&index=6"):
    
    final_chain = build_chatbot_chain(video_url)
    
    if not final_chain:
        return

    print("YouTube Video Q&A System")
    print("Ask questions about the video. Type 'quit' or 'exit' to stop.\n")
    print("-" * 63)

    while True:
        question = input("\nYour question: \n").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a valid question.")
            continue

        result = final_chain.invoke(question)
        print(f'\n{result}')

if __name__ == "__main__":
    youtube_chatbot()