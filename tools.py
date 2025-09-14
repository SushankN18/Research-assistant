from langchain_community.tools import WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Searches for a query on DuckDuckGo",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)