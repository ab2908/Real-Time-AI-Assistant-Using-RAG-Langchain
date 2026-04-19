from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# We specify the model we pulled in Step 1
llm = OllamaLLM(model="llama3:8b")
#llm = OllamaLLM(model="llama3.2:3b")

# This will run a web search
search = DuckDuckGoSearchRun()

# This is the prompt template, our instruction manual for the LLM
prompt = ChatPromptTemplate.from_template(
    """You are a helpful AI assistant. You must answer the user's question 
    based *only* on the following search results. If the search results 
    are empty or do not contain the answer, say 'I could not find 
    any information on that.'

    Search Results:
    {context}

    Question:
    {question}
    """
)

# This is our RAG chain
chain = (
    RunnablePassthrough.assign(
        context=lambda x: search.run(x["question"])
    )
    | prompt
    | llm     # The formatted prompt is "piped" into the LLM
)

print("🤖 Hello! I'm a real-time AI assistant. What's new?")
while True:
    try:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("🤖 Goodbye!")
            break
        
        print("🤖 Thinking...")
        
        # This one line runs the whole RAG process
        response = chain.invoke({"question": user_query})
        print(f"🤖: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")