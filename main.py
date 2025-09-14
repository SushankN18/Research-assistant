from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool

load_dotenv()

class Sus(BaseModel):
    topic:str
    summary:str
    sources: list[str]
    tools_used: list[str]



llm= ChatAnthropic(model="claude-3-5-sonnet-20241022")
parser = PydanticOutputParser(pydantic_object=Sus)

prompt = ChatPromptTemplate.from_messages(
[
    (
        "system",
        """
        You are a research assistant that will help genereate a research paper.
        Answer the user query and use necessary tools.
        Wrap the output in this format and provide no ther text\n{format_instructions}""",
    ),
    ("placeholder","{chat_history}"),
    ("human","{query}"),
    ("placeholder","{agent_scratchpad}"),
]
).partial(format_instructions = parser.get_format_instructions)

tools=[search_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)
agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
query = input("How can i assist you with your research? ")
raw_response = agent_executor.invoke({"query": query})
print(raw_response)

try:
    structured_repsone=parser.parse(raw_response.get("output")[0]["text"])
    print(structured_repsone)
except Exception as e:
    print("error is",e, "Raw Response",raw_response)
#