from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


tools = [TavilySearchResults(max_results=5, tavily_api_key="xxx")]

llm = ChatOpenAI(
    model="gpt-4o-2024-11-20",
    temperature=0,
    api_key="xxx",
)


builder = (
    StateGraph(MessagesState)
    .add_node(
        "chatbot",
        lambda state: {
            "messages": [
                (
                    ChatPromptTemplate.from_messages(
                        [
                            MessagesPlaceholder(variable_name="messages"),
                        ]
                    )
                    | llm.bind_tools(tools)
                ).invoke(state["messages"])
            ]
        },
    )
    .add_node("tools", ToolNode(tools))
    .set_entry_point("chatbot")
    .add_conditional_edges("chatbot", tools_condition)
    .add_edge("tools", "chatbot")
)

graph = builder.compile()

png = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png)
