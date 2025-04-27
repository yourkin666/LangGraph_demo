from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


tools = [
    TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=False,
        include_raw_content=False,
        include_images=False,
    )
]


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
                    | ChatOpenAI(
                        model="gpt-4o-2024-11-20",
                        temperature=0,
                    ).bind_tools(tools)
                ).invoke(state["messages"])
            ]
        },
    )
    .add_node(
        "tools",
        ToolNode(tools),
    )
    .set_entry_point("chatbot")
    .add_conditional_edges("chatbot", tools_condition)
    .add_edge("tools", "chatbot")
)

graph = builder.compile()
