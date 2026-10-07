import os
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from prompts import RAG_WORKFLOW_INSTRUCTIONS,SUBAGENT_DELEGATION_INSTRUCTIONS,CHUNK_ANALYST_INSTRUCTIONS
from tools import search_documentation,backend
from langchain_core.messages import HumanMessage

max_concurrent_analysis=3
INSTRCTIONS=(
    RAG_WORKFLOW_INSTRUCTIONS+"\n\n"+"="*80
    +"\n\n"
    +SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_analysts=max_concurrent_analysis
    )

)

CHUNK_ANALYST_subagent={
    "name":"chunks-analyst",
    "description":(
        "analyse one retrived documentation chunk file"
        "pass the use questions and a single file path under /retrived."
    ),
        "system_prompt":CHUNK_ANALYST_INSTRUCTIONS
}
model=init_chat_model(model="openai:gpt-5.5"
                      , api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    max_tokens=1000,
    )

agent=create_deep_agent(
    model=model,
    tools=[search_documentation],
    backend=backend,
    system_prompt=INSTRCTIONS,
    subagents=[CHUNK_ANALYST_subagent]
)

example_query = "where are penguins live?"

result = agent.invoke(
    {
        "messages": [
            ("user", example_query)
        ]
    }
)

print(result["messages"][-1].content)
print(result["messages"][-1].content)

