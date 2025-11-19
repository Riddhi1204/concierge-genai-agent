# concierge-genai-agent
# Concierge Agent Example (for Capstone & GitHub Deployment)

import os
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory

# Set your OpenAI API Key (recommended: use .env for secrets)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = OpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0.5,
    model="gpt-3.5-turbo"
)

# Define simple custom task tools
def welcome_tool(query):
    return "Welcome! How can I assist you today?"

def info_tool(query):
    return "I can help with scheduling, information lookup, and connecting you to specialized agents."

tools = [
    Tool(
        name="WelcomeTool",
        func=welcome_tool,
        description="Greets and introduces available services."
    ),
    Tool(
        name="InfoTool",
        func=info_tool,
        description="Explains what the concierge agent can do."
    ),
]

# Add session memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize Concierge Agent
concierge = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Example interaction loop (replace with Flask/FastAPI for deployment)
if __name__ == "__main__":
    print("Concierge Agent: Ready to serve. (Ctrl-C to exit)")
    while True:
        user_input = input("You: ")
        response = concierge.run(user_input)
        print("Agent:", response)
