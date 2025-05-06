from typing import Literal, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI


AGENT_NAME = "DebateAgent"

SUMMARY_SYSTEM_PROMPT = """You are a balanced debate analyst.
Analyze the following debate transcript, identify the key arguments from both the affirmative and negative sides, evaluate their strengths and weaknesses, and provide a concise, neutral summary of the discussion. Do not take a side.
Debate Topic: {topic}

Debate Transcript:
{transcript}

Provide your analysis and summary:"""


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
)


class DebateState(MessagesState):
    topic: Optional[str] = None


def get_topic(state: DebateState):
    """Extracts the debate topic from the initial human message."""
    initial_message = state["messages"][0]
    if isinstance(initial_message, HumanMessage):
        state["topic"] = initial_message.content
    else:
        state["topic"] = "General Discussion"
    return state


def generate_point(
    state: DebateState,
    side: Literal["Affirmative", "Negative"],
    action: Literal["Argument", "Rebuttal"],
) -> DebateState:
    """Generates an argument or rebuttal for the specified side using the LLM."""
    topic = state["topic"]
    messages_history = state["messages"]

    # 三元表达式
    prompt_action = (
        "provide the initial argument"
        if action == "Argument"
        else "provide a rebuttal to the previous point"
    )

    system_prompt = f"""You are participating in a debate on the topic: '{topic}'.
You are arguing for the {side.lower()} side.
Based on the conversation so far, {prompt_action}. Keep your response focused and concise."""

    return {"messages": [llm.invoke([SystemMessage(content=system_prompt)] + messages_history)]}


def affirmative_argument(state: DebateState) -> DebateState:
    return generate_point(state, "Affirmative", "Argument")


def negative_argument(state: DebateState) -> DebateState:
    return generate_point(state, "Negative", "Argument")


def rebuttal_negative(state: DebateState) -> DebateState:
    """Negative side rebuts the last point (which was Affirmative's argument/rebuttal)."""
    return generate_point(state, "Negative", "Rebuttal")


def rebuttal_affirmative(state: DebateState) -> DebateState:
    """Affirmative side rebuts the last point (which was Negative's argument/rebuttal)."""
    return generate_point(state, "Affirmative", "Rebuttal")


def summary_node(state: DebateState) -> DebateState:
    """Generates a summary of the debate using the LLM."""
    topic = state["topic"]
    transcript = "\n".join(
        [f"{msg.type.upper()}: {msg.content}" for msg in state["messages"][1:]]
    )

    summary_prompt = SUMMARY_SYSTEM_PROMPT.format(topic=topic, transcript=transcript)

    result = llm.invoke([SystemMessage(content=summary_prompt)])
    summary = result.content

    return {"messages": [SystemMessage(content=f"Summary:\n{summary}")]}


workflow = StateGraph(DebateState)

workflow.add_node("get_topic", get_topic)
workflow.add_node("affirmative_argument", affirmative_argument)
workflow.add_node("negative_argument", negative_argument)
workflow.add_node("rebuttal_by_negative", rebuttal_negative)
workflow.add_node("rebuttal_by_affirmative", rebuttal_affirmative)
workflow.add_node("summary", summary_node)

workflow.set_entry_point("get_topic")
workflow.add_edge("get_topic", "affirmative_argument")
workflow.add_edge("get_topic", "negative_argument")
workflow.add_edge("affirmative_argument", "rebuttal_by_affirmative")
workflow.add_edge("negative_argument", "rebuttal_by_negative")
workflow.add_edge("rebuttal_by_affirmative", "summary")
workflow.add_edge("rebuttal_by_negative", "summary")
workflow.add_edge("summary", END)


debate_agent = workflow.compile(checkpointer=MemorySaver())

