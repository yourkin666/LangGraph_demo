from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent


# Agent name
AGENT_NAME = "ContactAssist"

# System prompt for the chatbot
SYSTEM_PROMPT = """You are ContactAssist, a helpful assistant that can find information about people and send emails.

Your capabilities:
- Look up contact information for people by name
- Send emails to contacts

When users ask about someone, you can search for their information including name, email address, and position.
When users want to send an email, you can help them compose and send it to the recipient.

Always be professional, concise, and helpful in your responses.
"""


@tool
def get_person_info(name: str) -> list:
    """
    Get information about a person by name.

    Performs a fuzzy search by name and returns information about the person.

    Args:
        name: The name of the person to search for

    Returns:
        A list of dictionaries containing information about matching persons, including their email addresses
    """
    # Mock implementation using a list
    mock_data = [
        {"name": "John Smith", "email": "john.smith@example.com", "position": "Software Engineer"},
        {"name": "Jane Doe", "email": "jane.doe@example.com", "position": "Data Scientist"},
        {"name": "Alice Johnson", "email": "alice.johnson@example.com", "position": "Product Manager"},
        {"name": "Bob Brown", "email": "bob.brown@example.com", "position": "UI Designer"},
        {"name": "Sarah Johnson", "email": "sarah.johnson@example.com", "position": "Project Manager"},
        {"name": "Michael Johnson", "email": "michael.johnson@example.com", "position": "CTO"},
        {"name": "John Doe", "email": "john.doe@example.com", "position": "Developer"},
    ]


    # Perform fuzzy search and collect all matches
    matches = []

    for person in mock_data:
        if name.lower() in person["name"].lower():
            matches.append(person)

    # Return all matches
    return matches


@tool
def send_email(to: str, subject: str, body: str) -> bool:
    """
    Send an email to a specified recipient.

    Args:
        to: The email address of the recipient
        subject: The subject line of the email
        body: The content of the email

    Returns:
        True if the email was sent successfully, False otherwise
    """

    print(f"\nSending email to {to} with subject {subject} and body {body}\n")
    return True


tools = [get_person_info, send_email]

llm = ChatOpenAI(
    model="gpt-4o-2024-11-20",
    temperature=0,
    api_key="xxx",
)


manual_agent = (
    StateGraph(MessagesState)
    .add_node(
        "chatbot",
        lambda state: {
            "messages": [
                (
                    ChatPromptTemplate.from_messages(
                        [
                            SystemMessage(content=SYSTEM_PROMPT),
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
).compile(checkpointer=MemorySaver())

png = manual_agent.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png)

react_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
    checkpointer=MemorySaver(),
)
